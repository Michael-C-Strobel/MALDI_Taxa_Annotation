# Load the MALDIquant package
library(MALDIquant)
library(MALDIquantForeign)
library(compiler)

# Define the function
process_mzML_file <- function(input_file, output_file) {
    # Import the mzML file
    spectra <- importMzMl(input_file)
    
    # Smooth intensity
    spectra <- smoothIntensity(spectra, method="SavitzkyGolay", halfWindowSize=20L)
    
    # Remove baseline
    spectra <- removeBaseline(spectra, method="TopHat")
    
    # Detect peaks
    peaks = detectPeaks( spectra,
                         halfWindowSize=20L,
                         method="MAD",
                         SNR=4
                        )
    
    # Bin peaks
    peaks <- binPeaks(peaks, tolerance=0.001, method="strict")
    
    # Filter peaks that occur in less than 50% of replicates
    peaks <- filterPeaks(peaks, minFrequency=0.50)
    
    # Trim peaks
    peaks <- trim(peaks, c(2000, 20000))
    
    # Export to file
    exportMzMl(peaks, output_file, force=TRUE)
}

# Compile the function for optimization
process_mzML_file <- compiler::cmpfun(process_mzML_file)

# Execute with command-line arguments
args <- commandArgs(trailingOnly=TRUE)
process_mzML_file(args[1], args[2])
