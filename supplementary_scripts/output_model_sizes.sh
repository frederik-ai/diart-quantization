# https://superuser.com/questions/190653/ls-command-how-to-display-the-file-size-in-megabytes
function ls_with_size_in_MiB {
    ls -l | awk 'BEGIN{mega=1048576} {if($5 ~ /^[0-9]+$/) $5 = ($5 >= mega ? $5/mega " MiB" : $5 " bytes"); print}'
}

cd ..

cd baseline_models
echo "Baseline models:"
ls_with_size_in_MiB
cd ..

cd quantized_models
echo "Quantized models:"
ls_with_size_in_MiB