lines=( $(grep -E "outputB//tela-R-.*-1thr-exe.*" 00resultsPerf.csv) )
lines=( $(grep -E "outputB//tela-C-.*-1thr-exe.*" 00resultsPerf.csv) )
for ((i=0; i<${#lines[@]}; i+=2)); do  l1="${lines[i]}";  l2="${lines[i+1]}";  valores=$(echo -e $l1 "\n"$l2 |  awk -F"," '{printf(" %f %
f", $(NF-2), $(NF-1));} END{print " "}'); read nada t tp tu <<< "$valores"; echo  $tp / \($t - $tu \) +1 |bc ; done
