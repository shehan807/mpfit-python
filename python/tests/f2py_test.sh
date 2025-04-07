#!/bin/bash

# Define colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Directory paths
PYTHON_SCRIPT="/home/sparmar32/Desktop/scripts/mpfit-python/python/mpfit.py"
FORTRAN_EXEC="/home/sparmar32/Desktop/scripts/mpfit-python/f90/mpfit_source/source_linear_fit/mpfit"

# Function to test a single molecule
test_molecule() {
    local MOLECULE=$1
    echo "========================================"
    echo "Testing molecule: $MOLECULE"
    
    # Extract directory and filename
    DIR=$(dirname "$MOLECULE")
    FILENAME=$(basename "$MOLECULE")
    
    # Run Python version and capture output
    PYTHON_OUTPUT=$(python "$PYTHON_SCRIPT" "$MOLECULE")
    
    # Run Fortran version and capture output
    FORTRAN_OUTPUT=$("$FORTRAN_EXEC" "$MOLECULE")
    
    # Extract just the numeric values from each output
    PYTHON_VALUES=$(echo "$PYTHON_OUTPUT" | grep -E '^[A-Z]:' | awk '{print $2}')
    FORTRAN_VALUES=$(echo "$FORTRAN_OUTPUT" | grep -E '^[A-Z]' | awk '{print $2}')
    
    # Convert the outputs to arrays for comparison
    PYTHON_ARRAY=($PYTHON_VALUES)
    FORTRAN_ARRAY=($FORTRAN_VALUES)
    
    # Check if arrays have the same length
    if [ ${#PYTHON_ARRAY[@]} -ne ${#FORTRAN_ARRAY[@]} ]; then
        echo -e "${RED}ERROR: Different number of values in Python and Fortran outputs${NC}"
        echo "Python has ${#PYTHON_ARRAY[@]} values, Fortran has ${#FORTRAN_ARRAY[@]} values"
        return 1
    fi
    
    # Initialize comparison variables
    local ALL_MATCH=true
    local TOTAL_DIFF=0
    
    # Compare each value
    for i in "${!PYTHON_ARRAY[@]}"; do
        PYTHON_VAL=${PYTHON_ARRAY[$i]}
        FORTRAN_VAL=${FORTRAN_ARRAY[$i]}
        
        # Calculate absolute difference (using bc for floating point math)
        DIFF=$(echo "scale=10; v=($PYTHON_VAL)-($FORTRAN_VAL); if(v<0) -v else v" | bc)
        TOTAL_DIFF=$(echo "scale=10; $TOTAL_DIFF + $DIFF" | bc)
        
        # Check if difference is within tolerance (1e-5)
        if (( $(echo "$DIFF > 0.00001" | bc -l) )); then
            ALL_MATCH=false
            echo -e "${RED}Mismatch at value $i: Python=$PYTHON_VAL, Fortran=$FORTRAN_VAL, Diff=$DIFF${NC}"
        fi
    done
    
    # Print result for this molecule
    if [ "$ALL_MATCH" = true ]; then
        echo -e "${GREEN}✅ SUCCESS: All values match within tolerance${NC}"
        echo "Total cumulative difference: $TOTAL_DIFF"
        return 0
    else
        echo -e "${RED}❌ FAILURE: Values don't match within tolerance${NC}"
        echo "Total cumulative difference: $TOTAL_DIFF"
        return 1
    fi
}

# Main script starts here
TOTAL_SUCCESS=0
TOTAL_TESTS=0

# Check if molecules were provided as arguments
if [ $# -eq 0 ]; then
    # Default list of molecules to test if none provided
    MOLECULES=("acnit/acnit.dma" "sal/sal.dma")
else
    # Use molecules provided as arguments
    MOLECULES=("$@")
fi

# Run tests for each molecule
for molecule in "${MOLECULES[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    # Run the test for this molecule
    test_molecule "$molecule"
    
    # Track results
    if [ $? -eq 0 ]; then
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    fi
    
    echo ""
done

# Print summary
echo "========================================"
echo "SUMMARY: $TOTAL_SUCCESS/$TOTAL_TESTS tests passed"

# Exit with success only if all tests passed
if [ $TOTAL_SUCCESS -eq $TOTAL_TESTS ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
