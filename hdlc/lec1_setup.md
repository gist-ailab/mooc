1. Download Setup Files
    - Anaconda :: Windows 64 bit (https://www.anaconda.com/)
    - PyCharm :: Windows Community Version (https://www.jetbrains.com/pycharm/)

1. Install Anaconda (administrative)

1. Create an Environment in Anaconda (administrative)
     ```bash
    conda env list
    conda create --name test (conda create -n test)
    conda env list
    activate test
    deactivate
    conda env remove test
    ```

1. Install PyThon and PyTorch in Anaconda Environment
    ```bash
    conda create -n tc
    activate tc
    conda list
    conda install python
    conda list
    conda install pytorch-cpu torchvision-cpu -c pytorch
    deactivate
    ```

1. Run PyThon Code in Anaconda (administrative)
    - run python code in console mode
        ```bash
        activate tc
        python
        ```
        
        ```python
        print('hello world')
        import torch
        tc_version = torch.__version__
        print('torch version =',tc_version)      
        ```
    - save and run python code (test.py) in Anaconda Prompt 
        ```bash
        notepad => save "test.py"
        ```
        
        ```python
        import torch
        print('hello world')
        tc_version = torch.__version__
        print('torch version =',tc_version) 
        ```
        ```text
        dir
        cd Desktop
        python test.py
        ```
1. Install PyCharm (administrative)

1. Create a Project in PyCharm (administrative) based on Conda Environment

1. Run PyThon Code in PyCharm    
    - create a python file(test.py)
    - run the python code in Anaconda Prompt
    - run the python code in PyCharm
    - exectue line in PyCharm
    
1. Install PyThon Libraries: matplotlib, scipy, h5py
    - install libraries in Anaconda Prompt (administrative)
        ```bash
        activate tc
        conda install h5py matplotlib (pip install h5py matplotlib)
        conda uninstall h5py (pip uninstall h5py)        
        ```
    - install libraries in PyCharm (administrative)
        - install scipy via Conda Package Manager
        - install scipy via Pip Package Manager
        
1. Tutorials
    - Matplotlib Tutorials (https://matplotlib.org/tutorials/index.html)
    - SciPy Tutorials (https://docs.scipy.org/doc/scipy/reference/tutorial/)
    - h5py Tutorials (http://docs.h5py.org/en/stable/quick.html)
