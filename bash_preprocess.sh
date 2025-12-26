source ~/anaconda3/etc/profile.d/conda.sh
conda activate malegs

ulimit -n 10240

python preprocess_mv_aug_twoBranchUncertainly.py --dataset_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/trevi_fountain --appearance_target_path_list 15457887_10227170235 45182190_511249303 80288369_2336500045 --appearance_self_render True --iteration 350000 --itw_sh_degree 5 --itw_source_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/trevi_fountain --itw_model_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/trevi_fountain/trevi_fountain_wegs --pe_freq_xyz 4 

python preprocess_mv_aug_twoBranchUncertainly.py --dataset_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/buckingham_palace --appearance_target_path_list 04781012_3416228976 11220321_6429817645 42080522_204096736 --appearance_self_render True --iteration 250000 --itw_sh_degree 5 --itw_source_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/buckingham_palace --itw_model_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/buckingham_palace/buckingham_palace_wegs --pe_freq_xyz 4 

python preprocess_mv_aug_twoBranchUncertainly.py --dataset_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/pantheon_exterior --appearance_target_path_list 00318896_2265892479 02882184_6968792622 04938646_2803242734 --appearance_self_render True --iteration 250000 --itw_sh_degree 5 --itw_source_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/pantheon_exterior --itw_model_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/pantheon_exterior/pantheon_exterior_wegs --pe_freq_xyz 4 

python preprocess_mv_aug_twoBranchUncertainly.py --dataset_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/taj_mahal --appearance_target_path_list 75255818_297567547 76552970_5828212829 86954812_2533844894 --appearance_self_render True --iteration 200000 --itw_sh_degree 5 --itw_source_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/taj_mahal --itw_model_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/taj_mahal/taj_mahal_wegs --pe_freq_xyz 4 

python preprocess_mv_aug_twoBranchUncertainly.py --dataset_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/temple_nara_japan --appearance_target_path_list 08855480_12135166146 10449189_312833816 36907783_8321442187 --appearance_self_render True --iteration 200000 --itw_sh_degree 5 --itw_source_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/temple_nara_japan --itw_model_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/temple_nara_japan/temple_nara_japan_wegs --pe_freq_xyz 4 

python preprocess_mv_aug_twoBranchUncertainly.py --dataset_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/notre_dame_front_facade --appearance_target_path_list 73272860_3039447416 03158689_7322662838 72005271_4157221941 --appearance_self_render True --iteration 200000 --itw_sh_degree 5 --itw_source_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/notre_dame_front_facade --itw_model_path /media/wangyz/DATA/UBUNTU_data/dataset/PT/notre_dame_front_facade/notre_dame_front_facade_wegs --pe_freq_xyz 4 

