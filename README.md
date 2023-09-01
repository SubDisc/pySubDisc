# pySubDisc

pySubDisc is a Python wrapper for [SubDisc: Subgroup Discovery](https://github.com/SubDisc/SubDisc).

This package is still early in development and all functionality and syntax are subject to change. 

## Installation

This will be streamlined soon:

* From https://github.com/SubDisc/SubDisc, use `mvn package` to build `target/subdisc-gui.jar`
* Place `subdisc-gui.jar` in `src/pysubdisc/jars`
* Run `pip install .` from the root directory of the repository (containing pyproject.toml)

## Example

Using the data from https://github.com/SubDisc/SubDisc/blob/main/adult.txt :

```python
import pysubdisc
import pandas

data = pandas.read_csv('adult.txt')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasureMinimum = 0.25
sd.run()
print(sd.asDataFrame())
```

|    |   Depth |   Coverage |   Quality |   Target Share |   Positives |   p-Value | Conditions                            |
|---:|--------:|-----------:|----------:|---------------:|------------:|----------:|:--------------------------------------|
|  0 |       1 |        443 |  0.517601 |       0.440181 |         195 |       nan | marital-status = 'Married-civ-spouse' |
|  1 |       1 |        376 |  0.453305 |       0.446809 |         168 |       nan | relationship = 'Husband'              |
|  2 |       1 |        327 |  0.359959 |       0.428135 |         140 |       nan | education-num >= 11.0                 |
|  3 |       1 |        616 |  0.354077 |       0.334416 |         206 |       nan | age >= 33.0                           |
|  4 |       1 |        728 |  0.326105 |       0.311813 |         227 |       nan | age >= 29.0                           |
|  5 |       1 |        552 |  0.263425 |       0.317029 |         175 |       nan | education-num >= 10.0                 |

Some detailed examples can be found in the /examples folder.

## Documentation

The SubDisc documentation might be of help for working with pySubDisc: https://github.com/SubDisc/SubDisc/wiki.