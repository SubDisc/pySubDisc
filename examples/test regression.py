import pysubdisc
import pandas

data = pandas.read_csv('adult.txt')
sd = pysubdisc.singleNumericTarget(data, 'age') #using age as a (somewhat useless) target
sd.searchDepth = 1
sd.numericStrategy = 'BEST'
sd.qualityMeasure = 'EXPLAINED VARIANCE'

#run 100 swap-randomised SD runs in order to determine the minimum required quality to reach a significance level alpha = 0.05
sd.computeThreshold(setAsMinimum=True, verbose=False)
print("minimum quality for significance: ", sd.qualityMeasureMinimum)

#do the actual run and print the findings
sd.run()
print(sd.asDataFrame())
