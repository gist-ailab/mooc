1. Download Setup Files
    - Anaconda :: Windows 64 bit (https://www.anaconda.com/)
    - PyCharm :: Windows Community Version (https://www.jetbrains.com/pycharm/)

1. Install Anaconda (administrative)

1. Create an Environment via Anaconda Prompt (administrative)
     ```bash
    conda env list
    conda create --name test (conda create -n test)
    conda env list
    activate test
    deactivate
    conda env remove test
    ```

1. Install PyThon and PyTorch in Anaconda Environment via Anaconda Prompt (administrative)
    ```bash
    conda create -n practice
    activate practice
    conda list
    conda install python
    conda list
    deactivate
    exit
    ```

1. Run PyThon Code via Anaconda Prompt (administrative)
    - run python code in console mode
        ```bash
        activate practice
        python
        ```
        
        ```python
        print('hello world')
        import torch
        tc_version = torch.__version__
        print('torch version =',tc_version)      
        ```
    - save and run python code (test.py) via Anaconda Prompt 
        - create a file named "test.py" containing the python code below
            ```python
            import torch
            print('hello world')
            tc_version = torch.__version__
            print('torch version =',tc_version) 
            ```
        - run "test.py" in Anaconda Prompt 
            ```text
            dir
            cd Desktop
            python test.py
            ```
1. Install PyCharm (administrative)

1. Create "Practice" Project based on Conda Environment via PyCharm (administrative) 

1. Run PyThon Code in PyCharm    
    - create a python file(example_print.py)
    - run the python code in Anaconda Prompt
    - run the python code in PyCharm
    - exectue line in PyCharm

1. Install PyThon Libraries: matplotlib, scipy, h5py
    - install libraries via Anaconda Prompt (administrative)
        ```bash
        activate practice
        conda install matplotlib scipy (pip install h5py matplotlib)
        conda uninstall h5py (pip uninstall h5py)        
        ```
    - install libraries via PyCharm (administrative)
        - install and remove h5py via Pip Package Manager
        - install h5py via Conda Package Manager
            
    - create and run a python file(example_plot.py)
        - option scientific view (for each project)
    
    - create and run a python file(example_fft.py)
        
1. Create "PyTorch" Project in PyCharm (administrative) based on New Conda Environment named "tc"
    - install PyTorch via `Anaconda Prompt` (administrative)
        ```bash
        activate tc
        conda list
        conda install pytorch-cpu torchvision-cpu -c pytorch
        conda list
        ```    
    - install libraries (matplotlib, scify, h5py) via Conda Package Manager in `PyCharm` (administrative) 
    - create and run (example_torch.py)
        ```python
        import torch
        print(torch.__version__)
        print(torch.tensor([[1., -1.], [1., -1.]]))
        ```
        
1. Tutorials
    - Matplotlib Tutorials (https://matplotlib.org/tutorials/index.html)
    - SciPy Tutorials (https://docs.scipy.org/doc/scipy/reference/tutorial/)
    - h5py Tutorials (http://docs.h5py.org/en/stable/quick.html)
    - PyTorch Tutorials (https://pytorch.org/tutorials/)
