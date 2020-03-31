

##Running QC Model
for ss in CheXpert NIHDeepLesion PADCHEST; do

im=/data/DNNData/shared/COVID19Project/COVID_normal_images/$ss/
python qc_network.py --operation TEST --test_dir $im  --model_w /home/rs117/covid-19/models/model4-007-0.962333-0.863333-weights.h5 --out_csv ${ss}_normal_qc.csv

#Remapping label
sed -i 's/True/included/g' ${ss}_normal_qc.csv
sed -i 's/False/excluded/g' ${ss}_normal_qc.csv

done



#Moving data to each folder
dir=/data/DNNData/shared/COVID19Project/COVID_normal_images/

for ss in CheXpert NIHDeepLesion PADCHEST; do

aa=`cat ${ss}_normal_qc.csv`

  for bb in $aa; do
    ff=`echo $bb | cut -d , -f2`;
    tt=`echo $bb | cut -d , -f4`;
    sudo mv $dir/${ss}/$ff $dir/${ss}/$tt/;
  done

done

#Equialization
dir=/data/DNNData/shared/COVID19Project/COVID_normal_images/
out_dir=/data/DNNData/shared/COVID19Project/cxr_classification_consensus_normalized/

for ss in CheXpert; do
  mkdir ${out_dir}/${ss}/normal/

  aa=`ls -1 ${dir}/${ss}/included/*`

  for bb in $aa; do
    ff=`basename $bb`
    ff2=`echo $ff | cut -d . -f1`
    echo $bb
    echo ${out_dir}/$ss/normal/${ff2}.png
    python /home/rs117/src/devel/slowdown-covid19/equalization/equlize_cxr.py -i $bb -o ${out_dir}/$ss/normal/${ff2}.png
  done

done

for ss in NIHDeepLesion; do
mkdir ${out_dir}/${ss}/normal/

aa=`find ${dir}/${ss}/included/ -type f`

for bb in $aa; do
ff=`basename $bb`
echo $bb
echo ${out_dir}/$ss/normal/$ff
python /home/rs117/src/devel/slowdown-covid19/equalization/equlize_cxr.py -i $bb -o ${out_dir}/$ss/normal/$ff
done

done

for ss in PADCHEST; do
mkdir ${out_dir}/${ss}/normal/

aa=`ls -1 ${dir}/${ss}/included/*`

for bb in $aa; do
ff=`basename $bb`
echo $bb
echo ${out_dir}/$ss/normal/$ff
python /home/rs117/src/devel/slowdown-covid19/equalization/equlize_cxr.py -i $bb -o ${out_dir}/$ss/normal/$ff
done

done
