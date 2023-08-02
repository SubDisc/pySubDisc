# pySubDisc

This package is a Python wrapper for [SubDisc: Subgroup Discovery](https://github.com/SubDisc/SubDisc) .

pySubDisc is still early in development and all functionality and syntax are subject to change. 

## Installation

This will be streamlined soon:

* From https://github.com/SubDisc/SubDisc, use `mvn package -P lib` to build `target/subdisc-lib-2.1152.jar`.
* Place `subdisc-lib-2.1152.jar` in `src/pysubdisc/jars`
* Run `pip install .`

## Example

```python
import pysubdisc
import pandas

data = pandas.read_csv('adult.txt')
result = pysubdisc.runSubgroupDiscovery(data, 'single nominal', 'target', targetValue='gr50K')
print(result.asDataFrame())
```
