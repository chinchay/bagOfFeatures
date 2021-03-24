################################################################################
# Initialize

originalFile = "../testVac/data.lammps"
# originalFile = "/Users/cleon/Documents/projects/relax_pkg/testVac/inputLammpsOriginal"
cutOff = 6.0 #6.0


# get structure from file
include("myreadlammps.jl")
using Main.readLammpsModule
structure = getPymatgenStructFromLammpsInput(originalFile, "pymatgen")

# position of each id
fixedPositions = structure.cart_coords
# atomicFixedPositions = getAtomicFixedPositions(originalFile)

# neighbors of each id
allNeighbors = structure.get_all_neighbors(cutOff)
# getNeighborsPositions(atomIndex, allNeighborsFixed)

# just converting from Julia array to list:
n = size(fixedPositions)[1]
fixedPositions_ = [ [fixedPositions[i,1], fixedPositions[i,2], fixedPositions[i,3]] for i in 1:n ]

# get only coordinates information from PeriodicSite class
include("utils.jl")
allNeighbors_ = getCoordsAllNeighbors(allNeighbors)

size(allNeighbors_)
################################################################################

# Now structures with vacancies:
gamma           = 0.01
num_sites       = size(fixedPositions_)[1]
num_vacancies   = 18 ######################
dimFeatureSpace = 2 #1
n_clusters      = 4#4 #3#10 #3 # hyperparameter
dmin, dmax      = 0.5, cutOff

neighbors_indices_all = getNeighborsIndicesAll( structure, num_sites)


file_original_energy = "../testVac/energy_original.dat"
original_energy = getEnergy(file_original_energy)

fileEnergies         = "../testVac/inputsWithVacancies/energies_100"
energies = getEnergies(fileEnergies)
nStructures = size(energies)[1]

# instead of analyzing total energies, analyze change in the energy
# due to the vacancy:
# YOU MUST CONSIDER NOW Ga !!!!!@@@@@@@@@@@@
structureEnergies = energies - ones(nStructures)*original_energy
# nStructures = 1

################################################################################
# files = [ string(pwd(), "/testVac/inputsWithVacancies/data.lammps_", string(i) ) for i in 1:nStructures]
files = [ string("../testVac/inputsWithVacancies/data.lammps_", string(i) ) for i in 1:nStructures]
vacancyType = 7

# featMatrix = np.zeros( (nStructures * num_sites, dimFeatureSpace) )
featMatrix = zeros( (nStructures * num_vacancies, dimFeatureSpace) )
cont = 0

include("utils.jl")
featMatrix = getFeaturesFromAllStructs(files, vacancyType,
                                        fixedPositions_,
                                        allNeighbors_,
                                        nStructures, num_vacancies,
                                        dimFeatureSpace,
                                        dmin, dmax,
                                        neighbors_indices_all, cutOff)
assertMatrix(featMatrix, nStructures * num_vacancies, dimFeatureSpace)

################################################################################
include("utils.jl")
# featMatrix_julia = transpose(featMatrix)
# kmeans_julia, y_kmeans, centers = clustering(featMatrix, n_clusters,
#                                  nStructures, num_vacancies, dimFeatureSpace)
#
y_kmeans, objectkmeans, centers = pyClustering(featMatrix, n_clusters,
                                 nStructures, num_vacancies, dimFeatureSpace)

# for i in 1:size(y_kmeans)[1]
#     println(y_kmeans[i])
# end


y_kmeans_jl = getCorrected_y_kmeans(y_kmeans)
assertList(y_kmeans_jl, nStructures * num_vacancies)

plotClustering( featMatrix, y_kmeans, centers)


################################################################################
# population = uv.getPopulation(y_kmeans, n_clusters, nStructures, num_sites)
population = getPopulation(y_kmeans_jl, n_clusters, nStructures, num_vacancies)

assertMatrix( population, nStructures, n_clusters )

## y_kmeans = atomi_belongs2whichCluster
localEnergies = getLocalClusterEnergies(population, gamma,
                        n_clusters, structureEnergies, nStructures)

boolEmptyClusters = getBoolEmptyClusters(population, nStructures, n_clusters)


################################################################################

# include("utils.jl")
_, maxCluster = findmax(localEnergies)

# listIds = []
for iStruc in 1:5
    file = files[iStruc]
    # println(file)
    vacanciesIndx, _ = getVacanciesIndx(file, vacancyType) # begins in 0, python

    # y_kmeansReshaped = np.reshape( y_kmeans, (nStructures, num_vacancies) )
    # WARNING! Julia fills out by column, while Phython does by rows, which is
    # what we want:
    y_kmeansReshaped = reshape(y_kmeans_jl, (num_vacancies, nStructures)) |> transpose
    # y_kmeansReshaped = np.reshape( y_kmeans, (nStructures, num_vacancies) )
    y_kmeansForTheStructure = y_kmeansReshaped[iStruc, :]

    vacanciesIndxFilt = getAtoms2perturb(vacanciesIndx, y_kmeansForTheStructure,
                            maxCluster, num_vacancies)
    ids = vacanciesIndxFilt #+ 1
    # listIds.append()
    println(ids)
end



function getAproxEnergyCluster(localEnergies::Array, population::Array, iStruc::Int)
    aproxEnergy = 0
    for i in 1:3
        aproxEnergy +=localEnergies[i] * population[iStruc, i]
    end
    return aproxEnergy
end


x = zeros(nStructures)
y = zeros(nStructures)
for iStruc in 1:nStructures
    # iStruc = 0
    aproxEnergy = getAproxEnergyCluster(localEnergies, population, iStruc)
    # print(aproxEnergy, structureEnergies[iStruc])
    x[iStruc] = structureEnergies[iStruc]
    y[iStruc] = aproxEnergy
end
#
using Plots
scatter(x, y, legend=false, markersize = 10)


open("results", "w") do f
    for i in 1:size(x)[1]
        write(f, string(x[i]), " ", string(y[i]), "\n" )
    end
end


# scatter(x, y, legend=false, markersize = 10, add = raw"\draw (130, 130) -- (190,190)")
# scatter(x, y, add = raw"\draw (130, 130) -- (190,190)")
# add = raw"\draw (1,2) rectangle (2,3);"
# add = raw"\draw (130, 130) -- (190,190)"

# It didn't work
# PACKAGE: add PGFPlotsX
# pgfplotsx()
# plot(1:5, add = raw"\draw (1,2) rectangle (2,3);", extra_kwargs = :subplot)
