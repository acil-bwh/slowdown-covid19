data_dir=/data/DNNData/shared/COVID19Project/cxr_classification_consensus/COVID_project/
normal_dir=/data/DNNData/shared/COVID19Project/

normalized_dir=/data/DNNData/shared/COVID19Project/cxr_classification_consensus_normalized/

normal_dir=/data/DNNData/shared/COVID19Project/COVID_normal_images

out_dir=/data/DNNData/shared/COVID19Project/cxr_equalization

for ss in CheXpert NIHDeepLesion PADCHEST; do

  for tt in mild moderate-severe; do

    #Input file
    in_file_list=`find ${data_dir}/$ss/$tt/ | head -80`

    for ff in $in_file_list; do

    cp $ff ${out_dir}/input/

    gg=`basename $ff`

    root_name=`echo $gg | cut -d . -f1`

    cp ${normalized_dir}/$ss/$tt/${root_name}.png ${out_dir}/output/


    done


##out_file_list=`find ${normalized_dir}/$ss/$tt/ | head -80`
#
#    for ff in $out_file_list; do
#
#    cp $ff ${out_dir}/output/
#    done

 done

    in_file_list=`find ${normal_dir}/${ss}/included/ | head -140`
    for ff in $in_file_list; do

    cp $ff ${out_dir}/input/

    gg=`basename $ff`

    root_name=`echo $gg | cut -d . -f1`

    cp ${normalized_dir}/$ss/normal/${root_name}.png ${out_dir}/output/

    done


#  out_file_list=`find ${normalized_dir}/$ss/normal/ | head -140`
#
#  for ff in $out_file_list; do
#    cp $ff ${out_dir}/output/
#  done

done







