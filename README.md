# pySubDisc

pySubDisc is a Python wrapper for [SubDisc: Subgroup Discovery](https://github.com/SubDisc/SubDisc).

This package is still early in development and all functionality and syntax are subject to change. 

## Installation

This will be streamlined soon:

* From https://github.com/SubDisc/SubDisc, use `mvn package -P lib` to build `target/subdisc-lib-2.1152.jar`
* Place `subdisc-lib-2.1152.jar` in `src/pysubdisc/jars`
* Run `pip install .` from the root directory of the repository (containing pyproject.toml)

## Example

Using the data from https://github.com/SubDisc/SubDisc/blob/main/adult.txt :

```python
import pysubdisc
import pandas

data = pandas.read_csv('adult.txt')
result = pysubdisc.runSubgroupDiscovery(data, 'single nominal', 'target', targetValue='gr50K')
print(result.asDataFrame())
```

|    |   Depth |   Coverage |   Quality |   Target Share |   Positives |   p-Value | Conditions                            |
|---:|--------:|-----------:|----------:|---------------:|------------:|----------:|:--------------------------------------|
|  0 |       1 |        443 |  0.517601 |       0.440181 |         195 |       nan | marital-status = 'Married-civ-spouse' |
|  1 |       1 |        376 |  0.453305 |       0.446809 |         168 |       nan | relationship = 'Husband'              |
|  2 |       1 |        327 |  0.359959 |       0.428135 |         140 |       nan | education-num >= 11.0                 |
|  3 |       1 |        616 |  0.354077 |       0.334416 |         206 |       nan | age >= 33.0                           |
|  4 |       1 |        268 |  0.234734 |       0.38806  |         104 |       nan | hours-per-week >= 43.0                |
|  5 |       1 |        124 |  0.220187 |       0.548387 |          68 |       nan | occupation = 'Exec-managerial'        |
|  6 |       1 |        671 |  0.198276 |       0.28465  |         191 |       nan | sex = 'Male'                          |
|  7 |       1 |        166 |  0.193561 |       0.439759 |          73 |       nan | education = 'Bachelors'               |
|  8 |       1 |         50 |  0.17623  |       0.86     |          43 |       nan | capital-gain >= 4386.0                |
|  9 |       1 |        124 |  0.124776 |       0.41129  |          51 |       nan | occupation = 'Prof-specialty'         |
