#!/bin/bash

set -e

module unload python/2.7.18-ucs4
module load python/3.10.13-met
module load met

cd /scratch/ppatel/metplus_um_ens_pre_operational/common_input/masks

file_pp=/scratch/ppatel/metplus_um_ens_pre_operational/common_input/singv_eps/mss-aa981-ra3p2-DACE-202406/20240610T0600Z/um_000/umesga_pa000
var_pp=stratiform_rainfall_flux
var_pp_land=land_binary_mask
threshold_lat=">-4.8&&<7.1"
threshold_lon=">95.5&&<108.2"

# Extract SINGV domain exclude relax zones
echo "Extract SINGV domain exclude relax zones"
python3 ./convert_pp_to_nc.py $file_pp $var_pp

gen_vx_mask $var_pp.nc $var_pp.nc temp.nc -type "lat" -thresh $threshold_lat
gen_vx_mask temp.nc temp.nc singv_domain_mask.nc -type "lon" -thresh $threshold_lon -intersection -name "singv_domain"

# Extract land only
echo "Extract LAND mask"
python3 ./convert_pp_to_nc.py $file_pp $var_pp_land
gen_vx_mask singv_domain_mask.nc land_binary_mask.nc singv_land.nc \
    -type "data" -v 100 -mask_field 'name="land_binary_mask"; level="Surface";' \
    -thresh ">0.9" -intersection -name "land"

# Extract sea only
echo "Extract SEA mask"
gen_vx_mask land_binary_mask.nc land_binary_mask.nc  temp.nc \
    -type "data" -v 100 -mask_field 'name="land_binary_mask"; level="Surface";' \
    -thresh ">0.9" -complement

gen_vx_mask singv_domain_mask.nc temp.nc singv_sea.nc \
    -type "data" -v 100 -mask_field 'name="data_mask"; level="Surface";' \
    -thresh ">0.9" -intersection -name "sea"

# Extract SG only
#echo "Extract SG mask"
#gen_vx_mask singv_domain_mask.nc sg_shapefile/sg_boundary_parts_1.shp singv_sg.nc \
#    -type "shape" -v 100 -intersection -shape_str Name kml_1 -name "sg"
