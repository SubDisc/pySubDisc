def extraClassPath():
  import os
  # TODO: de-hardcode this name
  jar = 'subdisc-lib-2.1152.jar'
  jar = os.path.join(os.path.dirname(__file__), 'jars', jar)
  return [jar]

def ensureJVMStarted():
  import jpype
  import jpype.imports
  if not jpype.isJVMStarted():
    jpype.startJVM(classpath=extraClassPath())

  # Try to import a SubDisc class to raise an exception early if loading
  # SubDisc jar failed
  try:
    from nl.liacs.subdisc import Process
  except ModuleNotFoundError as e:
    # TODO: Python 3.11 has a cleaner approach using e.add_note()
    raise ModuleNotFoundError("Failed to import from SubDisc jar") from e
