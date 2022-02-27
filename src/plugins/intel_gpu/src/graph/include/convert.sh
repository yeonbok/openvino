for file in *.h;
do
#    if [ "$file" != "arg_max_min.cpp" ] && [ "$file" != "permute.cpp" ] && [ "$file" != "concat.cpp" ] && [ "$file" != "convert.sh" ];
#    then
#        git checkout 22fae50229be183631a70eb7c1692e4622dfc8f0 -- $file
#    fi
    mv $file $file.backup
done
