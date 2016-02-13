
module SB

using Distributions, JuMP, Gurobi

#data is one observation per row
#f(data) = zSAA, xSAA for that dataset
function baggedSAA(f, data, numBags)
	#solve one for one fun to get size of Xs
	z_temp, x_temp = f(data[1:2, :])
	
	xs_out = zeros(Float64, numBags, length(x_temp))
	zs_out= zeros(Float64, numBags)
	const n = length(x_temp)
	const N = size(data, 1)
	for iBag = 1:numBags
		zs_out[iBag], xs_out[iBag, :] = f(data[rand(1:N, N), :])
	end
	zs_out, xs_out
end


#SAA portfolio, CVaR constraint
function saa_port(eps_, budget, data)
	N, numAssets = size(data)

#	m = Model(solver=GurobiSolver(OutputFlag=false))
	m = Model()
	@defVar(m, xs[1:numAssets]>=0)
	@addConstraint(m, sum(xs) <= 1)
	@setObjective(m, Max, sum{dot(vec(data[j, :]), xs)/N, j=1:N})

	@defVar(m, Beta)
	@defVar(m, zs[1:N] >= 0)
	#define CvAr
	for j=1:N
		@addConstraint(m, -dot(vec(data[j, :]), xs[:])  - Beta <= zs[j])
	end
	@addConstraint(m, sum{zs[j]/eps_, j=1:N} + N*Beta <= N*budget)
	solve(m)
	getObjectiveValue(m) / N, getValue(xs[:])
end

function simMkt(numAssets, N)
    dat = randn(N, numAssets)
    dat = dat * diagm(linspace(.1, .3, numAssets))
    dat = broadcast(+, linspace(0, .1, numAssets)', dat)
end

#proxy CVaR by sorting
function cvar_sort(rets, eps_)
	const N = length(rets)
	rets = sort(rets) #makes a copy
	-mean(rets[1:floor(Integer, eps_ * N)])
end

##Simple Gaussian Mkt, CvaR portfolio
##Different amounts of initial data and bagging estimates
function test1(file_out, N_grid, numBags_grid, numRuns, seed=nothing)
	budget = .1
	eps_ = .1
	numAssets = 10
	out_mkt = simMkt(10, 50000)

	#Determine the benchmark optimal value
	#do this in a lazy way for now. 
	zt, xt = saa_port(eps_, budget, simMkt(10, 50000))
	z_out = mean(out_mkt*xt)
	cvar_out = cvar_sort(out_mkt*xt, eps_)

	f = open("$(file_out)_$(budget)_$(numAssets).csv", "w")
	writecsv(f, ["Run" "N" "numBags" "Method" "inReturn" "outReturn" "CVaR"])
	seed != nothing && srand(seed)

	for N in N_grid
		for numBags in numBags_grid
			for iRun = 1:numRuns
			    dat = SB.simMkt(numAssets, N)
			    z, x = SB.saa_port(eps_, budget, dat)
			    z_saa = mean(out_mkt * x) / z_out
			    cvar_saa = SB.cvar_sort(out_mkt*x, eps_) / budget
			    writecsv(f, [iRun N numBags "SAA" z z_saa cvar_saa])
	    
			    zt, xb = SB.baggedSAA(d-> SB.saa_port(eps_, budget, d), dat, numBags)
			    xb = vec(mean(xb, 1))
			    z_bag = mean(out_mkt * x) / z_out
			    cvar_bag = SB.cvar_sort(out_mkt*xb, eps_) /budget
			    writecsv(f, [iRun N numBags "Bag" mean(zt) z_bag cvar_bag])
			end
			flush(f)
		end
	end
	close(f)
end



end ##end module


