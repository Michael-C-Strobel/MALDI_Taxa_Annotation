# Load the MALDIquant package
library(MALDIquant)
library(MALDIquantForeign)

process_mzML_file <- function(input_file, output_file) {
    # Attempt to import the mzML file
    spectra <- importMzMl(input_file)
    # print("File imported successfully")

    # Print something 
    # print("Processing file")

    # smoothIntensity (commented out)
    spectra <- smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize = 20L)

    # removeBaseline
    spectra <- removeBaseline(spectra, method="SNIP", iterations=100)
    # print("Baseline removed")

    # detectPeaks
    peaks <- detectPeaks(spectra)
    # print("Peaks detected")

    # binPeaks
    peaks <- binPeaks(peaks, tolerance=0.002, method="strict")
    # print("Peaks binned")

    # filterPeaks
    peaks <- filterPeaks(peaks, minFrequency=0.50, minNumber=NA)

    # trim
    peaks <- trim(peaks, c(2000, 20000))
    print("Peaks trimmed")

    # Save to file
    exportMzMl(peaks, output_file, force=TRUE)
    print("Output saved to file")
    print(output_file)
}

args <- commandArgs(trailingOnly=TRUE)
process_mzML_file(args[1], args[2])