using Test # for @test

# using ScikitLearn
# @sk_import cluster.KMeans : KMEANs
### from sklearn.cluster import KMeans
using Clustering # we will use `kmeans`
using LinearAlgebra # to use `I` identity (to define identity matrix)

function getInteraction(A, B)
    # pair_coeff  *     *    0.0        1.0         0.0
    # pair_coeff  1     3    14509.63   0.2438      30.83         ## La - O
    # pair_coeff  2     3    2153.8     0.2908      0.0           ## Zr - O
    # pair_coeff  3     3    4869.99    0.2402      27.22         ## O  - O
    # pair_coeff  3     4    876.86     0.2433      0.0           ## Li4 - O
    # pair_coeff  3     5    876.86     0.2433      0.0           ## Li5 - O
    # pair_coeff  3     6    13298.83   0.2015      0.0           ## Ga - O
    #
    # Li-Ga --> Ga-Ga  ===> 1.0 to 1.0
    # Li-Ga --> Vac-Ga ===> 1.0 to 0
    # Li-Li --> Ga-Li  ===> 1.0 to 1.0
    # Li-Li --> Vac-Li ===> 1.0 to 0
    # Li-O  --> Ga-O   ===> 0.2433 to 0.2015 and 876.86 to 13298.83
    # Li-O  --> Vac-O  ===> 1.0 to 0
    # Li-Zr --> Ga-Zr  ===> 1.0 to 1.0
    # Li-Zr --> Vac-Zr ===> 1.0 to 0
    # Li-La --> Ga-La  ===> 1.0 to 1.0
    # Li-La --> Vac-La ===> 1.0 to 0
    #
    O   = 3
    # Li4 = 4
    # Li5 = 5
    Ga  = 6
    Vac = 7
    if (A == Vac) || (B == Vac)
        return 0
    elseif ( (A == Ga) && (B == O) ) || ( (A == O) && (B == Ga) )
        # println(A, "  ", B, " ", "Ga-O, or O-Ga")
        return 15.18
    else
        return 1
    end
end

function getWeight(j, i, atomTypes)
    return -1 * ( getInteraction(atomTypes[j], atomTypes[i]) - 1 )
end

# function getWeight(neighborsBody::Array, vacanciesIndx::Array, atomTypes::Array)
#     # neighborsBody is a list of indices representing the n-body
#     # neighborsBody does not contain the central atom, only neighbors
#     numberOfVacancies = 1 # we already have a vacancy (this program looks for them)
#     for j in neighborsBody
#         if is_j_vacancy(j, vacanciesIndx)
#             numberOfVacancies += 1
#         end
#     end
#     #
#     # if numberOfVacancies >= 1
#     #     weight = 0
#     # end
#     weight = 1.0 / numberOfVacancies # it will be =1/2 for a 2-body with 2 vacancies
#     # weight = 1.0
#     # #
#     # # if Ga is present in the n-body:
#     # for j in neighborsBody
#     #     type = atomTypes[j]
#     #     if type == 6 #typeX = 6 for Gallium
#     #         weight *= 15.18  # = 876.86/13298.83 = [3(Oxygen)---4(Li)] / [3(Oxygen)----6(Ga)]# see in.lammps2
#     #         # in fact you should visit Ga as well, not only vacancies.
#     #     end
#     # end
#     # #
#     # weight = 1
#     println(weight)
#     return weight
# end

function whereInt(array::Array, elementToFind::Int)
    listOfIndexes = []
    numberOfCoincidences = 0
    for (iterator, element) in enumerate(array)
        if element == elementToFind
            numberOfCoincidences += 1
            append!(listOfIndexes, iterator)
        end
    end
    return listOfIndexes, numberOfCoincidences
end

function getAtomTypes(file::String)
    start = false
    atomTypes = []
    for line in eachline(file)
        if start
            v = split(line)
            if size(v)[1] >= 1 # avoid blank lines
                atomType = parse(Int, v[3]) #
                append!(atomTypes, atomType)
            end
        end
        #
        if occursin( "Atoms", line )
            # print("found")
            # break
            start = true
        end
    end
    return atomTypes
end

function getVacanciesIndx2(file, vacancyType)
    # We are supposing every atom in the input file
    # has not been swapped, just changed of atomType, even
    # if it is now a vacancy
    atomTypes = getAtomTypes(file)
    vacancies, _ = whereInt(atomTypes, vacancyType) #7 is the vancacy type
    galliums, _ = whereInt(atomTypes, 6) #6 is the Ga type
    deffects = sort( append!(vacancies, galliums) )
    # println(deffects)
    return deffects, atomTypes
end

function getVacanciesIndx(file, vacancyType)
    # We are supposing every atom in the input file
    # has not been swapped, just changed of atomType, even
    # if it is now a vacancy
    atomTypes = getAtomTypes(file)
    vacancies, _ = whereInt(atomTypes, vacancyType) #7 is the vancacy type
    return vacancies, atomTypes
end


function belongs(x::Float64, xmin::Float64, xmax::Float64)
    return (xmin <= x) && (x <= xmax)
end

function getNorm(u1::Float64, u2::Float64, u3::Float64)
    return sqrt(u1^2 + u2^2 +  u3^2)
end

function substract(u1::Float64, u2::Float64, u3::Float64,
                    v1::Float64, v2::Float64, v3::Float64)
    return u1 - v1, u2 - v2, u3 - v3
end

function unit_vector(u1::Float64, u2::Float64, u3::Float64)
    """ Returns the unit vector of the vector v.  """
    norm_u = getNorm(u1, u2, u3)
    return u1/norm_u, u2/norm_u, u3/norm_u, norm_u
end

function dot(u1::Float64, u2::Float64, u3::Float64,
             v1::Float64, v2::Float64, v3::Float64)
    return u1 * v1 + u2 * v2 + u3 * v3
end

function clip(x::Float64, xmin::Float64, xmax::Float64)
    if x < xmin
        return xmin
    elseif xmax < x
        return xmax
    else
        return x
    end
end

function get_CosAngle_between(u1::Float64, u2::Float64, u3::Float64,
                              v1::Float64, v2::Float64, v3::Float64)
    u1, u2, u3, norm_u = unit_vector(u1, u2, u3)
    v1, v2, v3, norm_v = unit_vector(v1, v2, v3)
    return clip( dot(u1,u2,u3, v1,v2,v3), -1.0, 1.0 ), norm_u, norm_v
end

function fcut(d::Float64, dCut::Float64)
    if d <= dCut
        return 0.5 * (1 + cos(pi * d / dCut))
    else
        return 0
    end
end

function radialFeature(u1::Float64, u2::Float64, u3::Float64,
                       distMin::Float64, distMax::Float64, cutOff::Float64)
    eta_radial   = 0.05
    dCut  = 6.0
    #
    rij   = getNorm(u1, u2, u3)
    p     = (rij / dCut) ^ 2
    #
    return exp( (-eta_radial) * p ) * fcut(rij, dCut)

    # norm_u = getNorm(u1, u2, u3)
    # if belongs(norm_u, distMin, distMax)
    #     return 1
    # end
    # return 0
#
end

function angularFeature(u1::Float64, u2::Float64, u3::Float64,
                        v1::Float64, v2::Float64, v3::Float64,
                        cutOff::Float64)
    xi    = 1
    gamma = 1
    eta   = 0.005
    dCut  = cutOff #6.0 #8.0 #4.0
    #
    cos_phi, norm_u, norm_v = get_CosAngle_between(u1,u2,u3, v1,v2,v3)
    w1, w2, w3 = substract(u1,u2,u3, v1,v2,v3)
    norm_w = getNorm( w1, w2, w3 )
    #
    u2_ = norm_u ^ 2
    v2_ = norm_v ^ 2
    w2_ = norm_w ^ 2
    #
    t1 = 2.0 ^ (1 - xi)
    t2 = (1 - (gamma * cos_phi)) ^ xi
    t3 = exp( (-eta) * (u2_ + v2_ + w2_) / dCut )
    t4 = fcut(norm_u, dCut) * fcut(norm_v, dCut) * fcut(norm_w, dCut)
    #
    # println(t1 * t2 * t3 * t4, "<<<<")
    return t1 * t2 * t3 * t4
end


function is_j_vacancy(j::Int, vacanciesIndx::Array)
    return j in vacanciesIndx
end

function is_j_typeX(j::Int, typeX::Int, atomTypes::Array)
    return atomTypes[j] == typeX
end

function visit(iC, j, vacanciesIndx)
    visit = true
    if j in vacanciesIndx
        if j < iC
            # do not visit a vacancy already visited (avoid double counting)
            visit = false
        end
    end
    return visit
end

function getFeature_i2(  rCentral::Array,
                        neighbors::Array,
                        dimFeatureSpace::Int,
                        dmin::Float64,
                        dmax::Float64,
                        iC::Int,
                        vacanciesIndx::Array,
                        neighbors_indices_all::Array,
                        atomTypes::Array,
                        cutOff::Float64)
    feature = zeros(dimFeatureSpace)
    radial  = 0
    angular = 0
    neighbors_indices = neighbors_indices_all[iC]
    n = size(neighbors)[1]

    # if atomTypes[iC] == 6
    #     println("Ga", iC, " ", atomTypes[iC])
    #     println("Ga")
    # end

    # weight = 1
    for (j_enum, j) in enumerate(neighbors_indices)
        # if ( (atomTypes[iC] == 6)  && (atomTypes[j] == 3) )
        #     println("Ga and Oxygen: ", iC, " - ",j)
        # end
        # check if j is a vacancy, and if it was already visited
        if visit(iC, j, vacanciesIndx)
            # println(iC, "  -  ", j, "  -  ",  j in vacanciesIndx )
            rjC     = neighbors[j_enum] - rCentral
            # weight  = getWeight( j, iC, atomTypes )
            weight = 1
            if (j==153)
                weight = 15
            end

            # if ( (atomTypes[iC] == 6)  && (atomTypes[j] == 3) )
            #     println("Ga and Oxygen passed. weight=", weight)
            # end

            # weight = 1
            # weight  = getWeight( [j], vacanciesIndx, atomTypes )
            if weight != 0
                radial += weight * radialFeature(rjC[1], rjC[2], rjC[3], dmin, dmax, cutOff)
                # if weight != 1
                #     println(iC, " ", j, " ", atomTypes[iC], " ", atomTypes[j], " ", weight)
                # end
            end
            #
            for jj_enum in (j_enum + 1):n
                k = neighbors_indices[jj_enum]
                # check if k is a vacancy, and if it was already visited
                if visit(iC, k, vacanciesIndx)
                    rkC    = neighbors[jj_enum] - rCentral
                    # I don't know how is the interaction for 3-body ???
                    weight = 1 #getWeight( k, iC, atomTypes )
                    # weight   = getWeight( [j, k], vacanciesIndx, atomTypes )
                    angular += weight * angularFeature( rjC[1], rjC[2], rjC[3],
                                                        rkC[1], rkC[2], rkC[3], cutOff)
                end
            end
        # else
        #     println(iC, "  -  ", j, "  -  ",  )
        end
    end
    feature[1] = radial
    feature[2] = angular
    return feature
end

function getFeature_i(rCentral::Array, neighbors::Array,
                        dimFeatureSpace::Int, dmin::Float64, dmax::Float64,
                        iC::Int,
                        vacanciesIndx::Array,
                        neighbors_indices_all::Array,
                        atomTypes::Array,
                        cutOff::Float64)
                        # complete with zeros
                        # ( for radial and angular parts, dimFeatureSpace should be =2):
                        feature = zeros(dimFeatureSpace)

                        # radial function:
                        nNeighbors = size(neighbors)[1] # Julia begins in indx=1
                        radial  = 0
                        angular = 0

                        neighbors_indices = neighbors_indices_all[iC]

                        for j in 1:nNeighbors  # Julia begins in indx=1
                        # for j in range(nNeighbors):
                            # print("neighbors = ", len(neighbors))
                            # println(neighbors[j], " - ", size(neighbors), " - ", size(rCentral))



                            jNeighbor = neighbors_indices[j]

                            weight = 1.0
                            if is_j_typeX(jNeighbor, 6, atomTypes) #typeX = 6 for Gallium
                                # println( iC, " ---- ", jNeighbor )
                                weight = 15.18 # = 876.86/13298.83 = [3(Oxygen)---4(Li)] / [3(Oxygen)----6(Ga)]# see in.lammps2
                            end


                            # Avoid contribution if neighbor is another vacancy!
                            if !is_j_vacancy(jNeighbor, vacanciesIndx)
                                rjC = neighbors[j] - rCentral
                                radial += weight * radialFeature(rjC[1], rjC[2], rjC[3], dmin, dmax, cutOff)
                                # println(radial, " === ", j, " === ", rjC[1], " ",rjC[2], " ", rjC[3])
                                # println(neighbors[j])
                                # println(j, " ", rCentral, " ", nNeighbors)
                                for k in (j + 1):nNeighbors # avoid repeated (j,k) and (k,j)
                                    if (j != k) # avoid the same atom (However j is already != k)
                                    # if (j not in ) # DELETE THE OTHER VACANCIES, and the galliums?????????????????
                                        rkC = neighbors[k] - rCentral
                                        angular += weight * angularFeature(rjC[1],rjC[2],rjC[3], rkC[1],rkC[2],rkC[3], cutOff)

                                        # print("angular......................", angular)
                                    end
                                end
                            end
                        end
                        feature[1] = radial
                        feature[2] = angular
                        return feature

end



function getFeature_i3(rCentral::Array, neighbors::Array,
                        dimFeatureSpace::Int, dmin::Float64, dmax::Float64,
                        iC::Int,
                        vacanciesIndx::Array,
                        neighbors_indices_all::Array,
                        atomTypes::Array, cutOff::Float64)
    # complete with zeros
    # ( for radial and angular parts, dimFeatureSpace should be =2):
    feature = zeros(dimFeatureSpace)

    # radial function:
    nNeighbors = size(neighbors)[1] # Julia begins in indx=1
    radial  = 0
    angular = 0

    neighbors_indices = neighbors_indices_all[iC]

    for j in 1:nNeighbors  # Julia begins in indx=1
    # for j in range(nNeighbors):
        # print("neighbors = ", len(neighbors))
        # println(neighbors[j], " - ", size(neighbors), " - ", size(rCentral))



        jNeighbor = neighbors_indices[j]

        weight = 1.0
        if is_j_typeX(jNeighbor, 6, atomTypes) #typeX = 6 for Gallium
            # println( iC, " ---- ", jNeighbor )
            weight = 15.18 # = 876.86/13298.83 = [3(Oxygen)---4(Li)] / [3(Oxygen)----6(Ga)]# see in.lammps2

        end


        # Avoid contribution if neighbor is another vacancy!
        if !is_j_vacancy(jNeighbor, vacanciesIndx)
            rjC = neighbors[j] - rCentral
            radial += weight * radialFeature(rjC[1], rjC[2], rjC[3], dmin, dmax, cutOff)
            # println(radial, " === ", j, " === ", rjC[1], " ",rjC[2], " ", rjC[3])
            # println(neighbors[j])
            # println(j, " ", rCentral, " ", nNeighbors)
            for k in (j + 1):nNeighbors # avoid repeated (j,k) and (k,j)
                if (j != k) # avoid the same atom (However j is already != k)
                # if (j not in ) # DELETE THE OTHER VACANCIES, and the galliums?????????????????
                    rkC = neighbors[k] - rCentral
                    angular += weight * angularFeature(rjC[1],rjC[2],rjC[3], rkC[1],rkC[2],rkC[3], cutOff)

                    # print("angular......................", angular)
                end
            end
        end
    end
    feature[1] = radial
    feature[2] = angular
    return feature
end

function getFeatureStructure(vacanciesIndx::Array, fixedPositions::Array,
                            allNeighbors::Array, num_vacancies::Int,
                            dimFeatureSpace::Int, dmin::Float64, dmax::Float64,
                            neighbors_indices_all::Array,
                            atomTypes::Array,
                            cutOff::Float64)
    # assert num_sites == len(fixedPositions)

    # f = np.zeros( (num_sites, dimFeatureSpace) )
    f = zeros( num_vacancies, dimFeatureSpace )

    for (i, indx) in enumerate(vacanciesIndx)
        # print("i: ", i)
        rCentral  = fixedPositions[indx, :][1]

        neighbors = allNeighbors[indx]
        # println(size(fixedPositions), ",,,,,,,")

        # println(i, " ", indx, " ", fixedPositions[indx], " ",  rCentral, " - ", size(allNeighbors[indx][1]))

        v = getFeature_i(rCentral, neighbors, dimFeatureSpace, dmin, dmax, indx, vacanciesIndx, neighbors_indices_all, atomTypes, cutOff)
        f[i,1] += v[1]
        f[i,2] += v[2]
        # print('----------------------', i, f[i][1])
    end
    rows, cols = size(f)

    # assert rows == num_sites
    @test rows == num_vacancies
    @test cols == dimFeatureSpace
    return f
end

function getFeaturesFromAllStructs(files::Array, vacancyType::Int,
                                fixedPositions::Array,
                                allNeighbors::Array,
                                nStructures::Int, num_vacancies::Int,
                                dimFeatureSpace::Int,
                                dmin::Float64, dmax::Float64,
                                neighbors_indices_all::Array,
                                cutOff::Float64)
    # featMatrix = np.zeros( (nStructures * num_sites, dimFeatureSpace) )
    featMatrix = zeros( nStructures * num_vacancies, dimFeatureSpace )
    cont = 0

    for file in files
        vacanciesIndx, atomTypes = getVacanciesIndx(file, vacancyType) # begins in 0, python
        # print(vacanciesIndx)  # Indx != IDs !!! (Lammps begins in 1, python begins in 0)
        if cont != 0
            f = getFeatureStructure(vacanciesIndx, fixedPositions,
                                    allNeighbors, num_vacancies,
                                    dimFeatureSpace, dmin, dmax, neighbors_indices_all, atomTypes, cutOff)
            # featMatrix = np.row_stack( (featMatrix, f) )
            # hcat stands for horizontal concatenation, even though it's NOT horizontal !!
            featMatrix = vcat(featMatrix, f)

            # println("dimension: ", size(featMatrix))
        else
            featMatrix = getFeatureStructure(vacanciesIndx, fixedPositions,
                                    allNeighbors, num_vacancies,
                                    dimFeatureSpace, dmin, dmax, neighbors_indices_all, atomTypes, cutOff)
            # println("dimension: ", size(featMatrix))
        end
        cont += 1
    end

    # featMatrix = [
    #     getFeatureStructure(
    #                         getVacanciesIndx(file, vacancyType),
    #                         fixedPositions,
    #                         allNeighbors,
    #                         num_vacancies,
    #                         dimFeatureSpace,
    #                         dmin,
    #                         dmax
    #                         )
    #     for file in files
    # ]


    # println(featMatrix[:,2])
    rows, cols = size(featMatrix)
    # assert rows == nStructures * num_sites
    @test rows == nStructures * num_vacancies
    @test cols == dimFeatureSpace
    return featMatrix
end

function clustering(X::Array, n_clusters::Int,
                nStructs::Int, num_sites::Int,
                dimFeatureSpace::Int)
    # X = featMatrix
    # shape: (nStructs * num_sites, dimFeatureSpace)
    rows, cols = size(X)
    # allAtomsAllStructs, dimFeatureSpace = X.shape
    @test rows == nStructs * num_sites
    @test cols == dimFeatureSpace

    # R = kmeans(X, n_clusters; maxiter=200, display=:iter)
    Xt = transpose(X)
    result = kmeans(Xt, n_clusters; maxiter=200, display=:iter)
    @assert nclusters(result) == n_clusters # verify the number of clusters
    y_kmeans = assignments(result) # get the assignments of points to clusters
    # c = counts(result) # get the cluster sizes
    centers = result.centers # get the cluster centers

    @test size(y_kmeans)[1] == rows

    # atomi_belongs2whichCluster = y_kmeans
    # atomi_belongs2whichCluster is a matrix of int elements
    # of order nStructs x nAtoms
    # [fi=1, fi=2, f3, ..., f_nAtoms] --> [a, b, ... , c]
    # which says if atom_i belongs to the cluster `a`, or `c`, or ...
    return result, y_kmeans, centers
end

################################################################################
using PyCall
py"""
import numpy as np
from sklearn.cluster import KMeans

def clustering__( X, n_clusters: int,
                nStructs: int, num_sites: int,
                dimFeatureSpace: int):
    # X = featMatrix
    # shape: (nStructs * num_sites, dimFeatureSpace)
    rows, cols = X.shape
    # allAtomsAllStructs, dimFeatureSpace = X.shape
    assert rows == nStructs * num_sites
    assert cols == dimFeatureSpace

    # random_state=... random, but to make it reproducible
    kmeans = KMeans( n_clusters = n_clusters, max_iter=200, random_state=0 )
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    assert len(y_kmeans) == rows

    centers = kmeans.cluster_centers_

    # atomi_belongs2whichCluster = y_kmeans
    # atomi_belongs2whichCluster is a matrix of int elements
    # of order nStructs x nAtoms
    # [fi=1, fi=2, f3, ..., f_nAtoms] --> [a, b, ... , c]
    # which says if atom_i belongs to the cluster `a`, or `c`, or ...
    # return kmeans, y_kmeans
    return y_kmeans, kmeans, centers
#
"""
pyClustering(featMatrix, n_clusters, nStructures, num_vacancies, dimFeatureSpace) = py"clustering__"(featMatrix, n_clusters, nStructures, num_vacancies, dimFeatureSpace)

function getCorrected_y_kmeans(y_kmeans)
    # this function corrects the fact that Python begins in 0, while Julia, in 1
    n = size(y_kmeans)[1]
    y_kmeans_julia = zeros(Int, n)
    for i in 1:n
        # remember, it is an array of indexes (of iClusters)
        y_kmeans_julia[i] = y_kmeans[i] + 1 # translated to Julia's indexes
    end
    #
    return y_kmeans_julia
end

################################################################################


function getPopulation(y_kmeans::Array, n_clusters::Int,
                nStructs::Int, num_sites::Int)
    # population is the bag of features for all structures: matrix
    # each row is associated with a structure
    # For each row:
    # [fi=1, fi=2, f3, ..., f_nAtoms] --> [ N1, N2, ..., NC ]
    # in such a way that N1 + N2 + ... + NC = nAtoms
    #
    @test size(y_kmeans)[1] == nStructs * num_sites

    # Y = np.reshape(y_kmeans, (nStructs, num_sites))
    # WARNING! Julia fills out by column, while Phython does by rows, which is
    # what we want:
    Y = reshape(y_kmeans, (num_sites, nStructs)) |> transpose
    # rows, cols = Y.shape()
    population = zeros( Int, nStructs, n_clusters )
    for row in 1:nStructs
        y = Y[row, :]
        numberOfCoincidences = 0
        for iCluster in 1:n_clusters
            # # tempArray = np.where(y == c)
            # for e in y
            #     if e == c
            #         numberOfCoincidences += 1
            #     end
            # end
            _, numberOfCoincidences = whereInt(y, iCluster)
            population[row, iCluster] = numberOfCoincidences
        end
    end
    for row in 1:nStructs
        @test sum(population[row, :]) == num_sites
    end
    return population
end

function minimize(X::Array, E::Array, gamma::Float64)
    # Solves: [X][sol] = [E], see paper
    _, cols = size(X)
    Xt = transpose(X)

    # I = np.identity(cols)
    gammaIdentity = Matrix(gamma*I, cols, cols)

    temp = inv( ( Xt * X ) + gammaIdentity )
    sol = temp * Xt * E
    #
    return sol
end

function getLocalClusterEnergies(population::Array, gamma::Float64,
                    n_clusters::Int, strucEnergies::Array,
                    nStructs::Int)

    # population is the bag of features for all structures: matrix
    # Solve: population . X = Column of Energies known
    # localEnergies = np.zeros(n_clusters)

    rows, cols = size(population)
    @test rows == size(strucEnergies)[1]
    @test rows == nStructs
    @test cols == n_clusters

    # localEnergies = np.linalg.solve(population, strucEnergies)
    localEnergies = minimize(population, strucEnergies, gamma)
    @test size(localEnergies)[1] == n_clusters
    return localEnergies
end

function getBoolEmptyClusters(population::Array, nStructs::Int, n_clusters::Int)
    # emptyCluster = -1
    rows, cols = size(population)
    @test rows == nStructs
    @test cols == n_clusters

    columOfZeros = zeros( nStructs )
    boolEmptyClusters = falses(n_clusters)
    for iCluster in 1:n_clusters
        pop_iCluster = population[:, iCluster]

        # boolResult = np.all( pop_iCluster == columOfZeros )
        boolResult   = ( pop_iCluster == columOfZeros )

        boolEmptyClusters[iCluster] = boolResult
    end
    return boolEmptyClusters # array of bools
end


function getCoordsAllNeighbors(allNeighbors)
    n = size(allNeighbors)[1]
    allNeighborsCoords = [
        [
            [
                allNeighbors[i][j].coords[1]
                allNeighbors[i][j].coords[2]
                allNeighbors[i][j].coords[3]
            ]
            for j in 1:( size(allNeighbors[i])[1] )
        ]
        for i in 1:n
    ]
    #
    return allNeighborsCoords
end

function getEnergies(fileEnergies)
    energies = []
    for line in eachline(fileEnergies)
        v = split(line)
        append!( energies, parse(Float64, v[1]) )
    end
    return energies
end

function getEnergy(fileEnergy)
    v = []
    for line in eachline(fileEnergy)
        v = split(line)
    end
    return parse(Float64, v[1])
end


function assertMatrix(matrix::Array, m::Int, n::Int)
    rows, cols = size(matrix)
    @test rows == m
    @test cols == n
end

function assertList(L::Array, lenL::Int)
    @test size(L)[1] == lenL
end


using Plots
function plotClustering(X::Array, y_kmeans::Array, centers::Array)
    # plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    # # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

    # result = kmeans(features, 3); # run K-means for the 3 clusters

    # plot with the point color mapped to the assigned cluster index
    # scatter(X[:, 1], X[:, 2], marker_z=y_kmeans, color=:lightrainbow, legend=false)
    scatter(X[:, 1], X[:, 2], marker_z=y_kmeans, colo=:viridis, legend=false, markersize = 10)

end


function getAtoms2perturb(vacanciesIndx::Array,
                    y_kmeansForTheStructure::Array,
                    maxCluster::Int,
                    num_vacancies::Int)
    # returns a list of indexes, which coincides with the atom index!
    @test size(y_kmeansForTheStructure)[1] == num_vacancies
    # atoms2perturb = np.where(y_kmeansForTheStructure == maxCluster)[0]
    indexes, _ = whereInt(y_kmeansForTheStructure, maxCluster)
    return vacanciesIndx[indexes]
end


function getNeighborsIndicesAll( structure, num_sites)
    # https://pymatgen.org/pymatgen.core.structure.html
    center_indices, neighbors_indices, _, distances = structure.get_neighbor_list(cutOff)

    # println( "len = ", size(neighbors_indices)[1])

    neighbors_indices_all = []
    distances_all = []
    for i in 1:num_sites
        iPython = i - 1
        columns = whereInt(center_indices, iPython)[1]
        # columnsPython = [ col + 1 for col in columns ]

        neighbors_indices_iAtom = []
        distances_iAtom = []
        for col in columns
            # println("i, col = ", i, " ", col)
            # add `1` for Julia indexing
            append!( neighbors_indices_iAtom, neighbors_indices[col] + 1 )
            append!( distances_iAtom, distances[col] )
        end
        append!( neighbors_indices_all, [neighbors_indices_iAtom] )
        append!( distances_all, distances_iAtom )
    end

    return neighbors_indices_all
end
