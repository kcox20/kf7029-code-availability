from RunModel import RunModel

RunModel("data/breast-cancer-GSE75688/TPM_matrix_tumor_single_cells_GSE75688.csv",
         #use_huang_2023_args=True,
         lr=1e-3,
         parallel_layers=["mod2", "AR"],
         batching=True,
         epochs=60,
         clusters_file_name="result/original_args_excl_module_3_dataset_2_clusters_60_epochs.csv")
