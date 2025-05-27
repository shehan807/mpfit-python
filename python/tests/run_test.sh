#echo "Running Test for acnit.dma"
#echo "%%% PYTHON CHARGES %%%"
#MOL="acnit"
#python /home/sparmar32/Desktop/scripts/mpfit-python/python/mpfit.py\
#       ${MOL}/${MOL}.dma
#echo "%%% FORTRAN CHARGES %%%"
#/home/sparmar32/Desktop/scripts/mpfit-python/f90/mpfit_source/source_linear_fit/mpfit\
#       ${MOL}/${MOL}.dma
#
#echo "Running Test for sal.dma"
#echo "%%% PYTHON CHARGES %%%"
#MOL="sal"
#python /home/sparmar32/Desktop/scripts/mpfit-python/python/mpfit.py\
#       ${MOL}/${MOL}.dma
#echo "%%% FORTRAN CHARGES %%%"
#/home/sparmar32/Desktop/scripts/mpfit-python/f90/mpfit_source/source_linear_fit/mpfit\
#       ${MOL}/${MOL}.dma

echo "Running Test for water.dma"
echo "%%% PYTHON CHARGES %%%"
MOL="water"
python /home/sparmar32/Desktop/scripts/mpfit-python/python/mpfit.py\
       ${MOL}/${MOL}.dma
echo "%%% FORTRAN CHARGES %%%"
/home/sparmar32/Desktop/scripts/mpfit-python/f90/mpfit_source/source_linear_fit/mpfit\
       ${MOL}/${MOL}.dma
