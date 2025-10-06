from ete3 import NCBITaxa
import pandas as pd
import requests
import json
import pytest
from time import sleep
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import traceback
import sys

@retry(
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((
        requests.exceptions.RequestException,
        requests.exceptions.ConnectTimeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    )),
    reraise=True,
)
def fetch_with_retry(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

def get_ncbi_taxid_from_genbank(genbank_accession:str)->int:
    """Gets the NCBI taxid from a genbank accession. Each genbank accession is
    checked in the nucleotide and genome databases (priority is given to nucleotide).
    
    Args:
        genbank_accession (str): The genbank accession number
        
    Returns:
        int: The NCBI taxid. Empty string if not found.
    """

    genbank_accession = str(genbank_accession)

    # First check the nucleotide database
    nucleotide_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=nucleotide&term={genbank_accession}&retmode=json"
    r = fetch_with_retry(nucleotide_url)
    nucleotide_json = json.loads(r, strict=False)
    # print("nucleotide_json", nucleotide_json, flush=True)

    nucleotide_taxid = ""
    if "esearchresult" in nucleotide_json:
        if "idlist" in nucleotide_json["esearchresult"]:
            if len(nucleotide_json["esearchresult"]["idlist"]) > 0:
                nucleotide_id = nucleotide_json["esearchresult"]["idlist"][0]
                nucleotide_summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nucleotide&id={nucleotide_id}&retmode=json"
                r = fetch_with_retry(nucleotide_summary_url)
                nucleotide_summary_json = json.loads(r, strict=False)
                if "result" in nucleotide_summary_json:
                    if nucleotide_id in nucleotide_summary_json["result"]:
                        if "taxid" in nucleotide_summary_json["result"][nucleotide_id]:
                            nucleotide_taxid = nucleotide_summary_json["result"][nucleotide_id]["taxid"]

    # If not found in nucleotide, check the assembly database
    if nucleotide_taxid == "":
        assembly_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=assembly&term={genbank_accession}&retmode=json"
        r = fetch_with_retry(assembly_url)
        assembly_json = json.loads(r, strict=False)

        if "esearchresult" in assembly_json:
            if "idlist" in assembly_json["esearchresult"]:
                if len(assembly_json["esearchresult"]["idlist"]) > 0:
                    assembly_id = assembly_json["esearchresult"]["idlist"][0]
                    assembly_summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={assembly_id}&retmode=json"
                    r = fetch_with_retry(assembly_summary_url)
                    assembly_summary_json = json.loads(r, strict=False)

                    if "result" in assembly_summary_json:
                        if assembly_id in assembly_summary_json["result"]:
                            if "taxid" in assembly_summary_json["result"][assembly_id]:
                                nucleotide_taxid = assembly_summary_json["result"][assembly_id]["taxid"]

    return nucleotide_taxid

def get_taxonomy_dict_from_ncbi(taxid:int, ncbi_taxa:NCBITaxa):
    """Gets the taxonomic linear as a dictionary using a taxid. 
    
    Args:
        taxid (str): The taxid of the organism
        ncbi_taxa (NCBITaxa): The ncbi taxonomy database
        
    Returns:
        dict: The taxonomic lineage dictionary
    """
    taxid = int(taxid)
    if not isinstance(ncbi_taxa, NCBITaxa):
        raise ValueError("ncbi_taxa must be an instance of NCBITaxa")
    
    lineage_as_int = ncbi_taxa.get_lineage(taxid)
    # print("Lineage as int", lineage_as_int, flush=True)
    names = ncbi_taxa.get_taxid_translator(lineage_as_int)
    ranks = ncbi_taxa.get_rank(lineage_as_int)
    common_keys = list(set(names.keys()).intersection(set(ranks.keys())))

    lineage_dict = {}
    for key in common_keys:
        if ranks[key] != 'no rank':
            lineage_dict[ranks[key]] = names[key]

    return lineage_dict

def get_taxonomy(spectra_entry, ncbi_taxa):
    """Gets the taxonomic lineage string for a spectra entry. First uses the genbank
    accession to get the lineage. If the genbank accession is not available, the
    NCBI taxid is used as a fallback. If both are unavailable, an empty string is

    Args:
        spectra_entry (dict): The spectra database entry.
        ncbi_taxa (NCBITaxa): The ncbi taxonomy database

    Returns:
        str: The taxonomic lineage string. Empty string if there is an error.
    """
    taxonomy_dict = {}
    genbank_accession = spectra_entry.get("Genbank accession", "")

    ncbi_taxid = spectra_entry.get("NCBI taxid", "")

    if genbank_accession != "" and genbank_accession != "None" and not pd.isna(genbank_accession):
        # Prefer genbank over NCBI taxid
        try:
            ncbi_taxid = get_ncbi_taxid_from_genbank(genbank_accession)
        except Exception as _:
            print("Exception while getting NCBI taxid for Genbank accession", genbank_accession, flush=True)
            traceback.print_exc(file=sys.stdout)
            pass

    if ncbi_taxid != "" and ncbi_taxid != "None" and not pd.isna(ncbi_taxid):
        # Use the given NCBI taxid as a fallback
        try:
            ncbi_taxid = int(ncbi_taxid)
            taxonomy_dict = get_taxonomy_dict_from_ncbi(ncbi_taxid, ncbi_taxa)

        except Exception as e:
            print("Exception while getting taxonomy for NCBI taxid", ncbi_taxid, flush=True)
            traceback.print_exc(file=sys.stdout)
            pass

        if taxonomy_dict != {}:
            return taxonomy_dict, ncbi_taxid
        else:
            return {}, None
        
def populate_taxonomies(summary_df: pd.DataFrame):
    """Populates the FullTaxonomy and final NCBI taxid for each entry in the dataframe.
    
    Args:
        summary_df (pd.DataFrame): The summary dataframe associated with the dataset.
        
    Returns:
        pd.DataFrame: The dataframe with added columns for taxonomy.
    """

    # ncbi_taxa = NCBITaxa(update=False)
    # ncbi_taxa.update_taxonomy_database()

    # Deduplicate to reduce redundant NCBI queries
    non_duplicated = summary_df.drop_duplicates(subset=["Genbank accession", "NCBI taxid"])
    tax_dict = non_duplicated[["Genbank accession", "NCBI taxid", "16S Taxonomy"]].to_dict(orient="records")

    enriched_entries = []
    for entry in tqdm(tax_dict):
        try:
            taxonomy_dict, ncbi_taxid = get_taxonomy(entry, ncbi_taxa)
            entry["FullTaxonomy"] = taxonomy_dict
            entry["ResolvedTaxid"] = ncbi_taxid
            for key, value in entry["FullTaxonomy"].items():
                if value is not None:
                    entry[key] = value
        except Exception:
            entry["FullTaxonomy"] = {}
            entry["ResolvedTaxid"] = None
            traceback.print_exc(file=sys.stdout)
        enriched_entries.append(entry)

    print(enriched_entries)

    tax_df = pd.DataFrame(enriched_entries)
    print(tax_df)

    summary_df = summary_df.merge(
        tax_df,
        how="left",
        on=["Genbank accession", "NCBI taxid", "16S Taxonomy"]
    )
    return summary_df

@pytest.fixture
def taxonomy_entry():
    data = {
        'Genbank accession': ['NC_000913.3'],
        'NCBI taxid': ['511145'],
        '16S Taxonomy': ['Escherichia coli'],
        'FullTaxonomy': ['Bacteria;Proteobacteria;Gammaproteobacteria;Enterobacterales;Enterobacteriaceae;Escherichia;Escherichia coli']
    }
    
    df = pd.DataFrame(data)
    return df

def test_populate_from_genbank(taxonomy_entry):
    ncbi_taxa = NCBITaxa('../data/ncbi_taxa')
    ncbi_taxa.update_taxonomy_database()

    _taxonomy_entry = taxonomy_entry.copy()
    _taxonomy_entry.drop(columns=['FullTaxonomy'], inplace=True)
    _taxonomy_entry['NCBI taxid'] = ""

    recovered_entry = populate_taxonomies(_taxonomy_entry)

    print(recovered_entry)

    assert recovered_entry['NCBI taxid'].iloc[0] == "511145", f"Expected NCBI taxid 511145, got {recovered_entry['NCBI taxid'].iloc[0]}"
    assert recovered_entry['FullTaxonomy'].iloc[0] == taxonomy_entry['FullTaxonomy'].iloc[0], f"Expected FullTaxonomy {taxonomy_entry['FullTaxonomy'].iloc[0]}, got {recovered_entry['FullTaxonomy'].iloc[0]}" 