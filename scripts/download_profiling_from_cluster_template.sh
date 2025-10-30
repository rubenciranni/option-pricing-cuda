# echo in red
RED='\033[0;31m'
NC='\033[0m' # No Color
echo -e "${RED}WARNING: This is a template script for profiling. Modify it as needed.${NC}"

## sytax scp
# scp username@cluster_address:/path/to/remote/file /path/to/local/destination

source ./scripts/.credentials
if [ -z "$USERNAME" ] || [ -z "$HOST" ]; then
    echo -e "${RED}ERROR: Please set USERNAME and HOST in scripts/.credentials file.${NC}"
    exit 1
fi

scp ${USERNAME}@${HOST}:/home/${USERNAME}/option-pricing-cuda/profile_res/* /tmp/profile_res/


nsys-ui /tmp/profile_res/profile_report.nsys-rep
ncu-ui /tmp/profile_res/profile_kernel.ncu-rep