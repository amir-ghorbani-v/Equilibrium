#!/bin/bash
declare -a PotentialArray=("M2" "M3" "BMD192" "M2R" "M3R" "BMD192R")

csv_file="/scratch/veshand/Zr/Simulation/Equilibrium/Mapping/20240228-Equilibrium.csv"

for Potential in "${PotentialArray[@]}"
	do
		echo "Processing Potential: $Potential"

		A1X=$(awk -F, -v p="$Potential" '$1 == p { print $3 }' "$csv_file")
		A2Y=$(awk -F, -v p="$Potential" '$1 == p { print $4 }' "$csv_file")
		A3Z=$(awk -F, -v p="$Potential" '$1 == p { print $5 }' "$csv_file")

		mkdir -p "$Potential"
		cd "$Potential" || exit

		sed -e "s/PotentialTemp/$Potential/g" ../20250130-Equilibrium.job > 20250130-Equilibrium.job
		sed -e "s/PotentialTemp/$Potential/g; s/BoxXTemp/$A1X/g; s/BoxYTemp/$A2Y/g; s/BoxZTemp/$A3Z/g" ../20250130-Equilibrium.lammpstemp > 20250130-Equilibrium.lammpsin

		sbatch 20250130-Equilibrium.job
		#chmod +rwx "20250130-Equilibrium.job"
		#./20250130-Equilibrium.job
		cd ..
    done


echo "All jobs submitted successfully."
