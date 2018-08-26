As of release 2017.1 PyCharm supports using Docker containers as remote interpreters.
However even as of 2018.1.4 there are some quirks related to the setup.

1. In Preferences -> Build/Execution/Deployment configure your Docker daemon.
    1. Remove any path mappings present by default.
    2. Ensure that PyCharm reports "connection successful"
2. In Preferences -> Project: $name_of_this_project -> Project Interpreter add the docker remote interpreter.
    1. Click the gear icon -> Add -> Docker
    2. Image name: `derivedjupyter:latest`
    3. Interpeter path: `python`
    4. Ensure that PyCharm picks up the installed modules.
    5. Add the path mapping from local `/$ABS_PATH_TO/empirical_privacy` to `/emp_priv` in the container.
3. In Run -> Edit Configurations -> wrench and nut icon edit default run configurations
    1. For both for `Python` and `Python Tests/py.test`:
        1. Ensure the mapping `-v /$ABS_PATH_TO/empirical_privacy:/emp_priv` is in Docker Container settings.
    2. For `Python` disable "Run in Python Console" source: https://youtrack.jetbrains.com/issue/PY-28608 .