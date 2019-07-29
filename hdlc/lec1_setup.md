1. Download Setup Files
    - Anaconda :: Windows 64 bit (https://www.anaconda.com/)
    - PyCharm :: Windows Community Version (https://www.jetbrains.com/pycharm/)

1. Install Anaconda (administrative)

1. Create an Environment via **Anaconda Prompt (administrative)**
     ```bash
    conda env list
    conda create --name test
    conda env list
    activate test
    deactivate
    conda env remove -n test
    ```

1. Install PyThon in Anaconda Environment via **Anaconda Prompt (administrative)**
    ```bash
    conda create -n practice
    activate practice
    conda list
    conda install python
    conda list
    deactivate
    exit
    ```

1. Run PyThon Code via **Anaconda Prompt (administrative)**
    - run python code in console mode via **Anaconda Prompt (administrative)**
        ```bash
        activate practice
        python
        ```
        
        ```python
        print('hello world')
        n1, n2 = 1.7, 1.5
        sum = n1 + n2
        print('The sum of {0} and {1} is {2}'.format(n1, n2, sum))
        print('The sum of {} and {} is {}'.format(n1, n2, sum))
        print('The sum of {:.0f} and {:05.2f} is {:5.2f}'.format(n1, n2, sum))
        ```
    - save and run python code (ex_sum.py) via **Anaconda Prompt (administrative)**
        - create a python file (c:\PycharmProjects\example\ex_sum.py) containing the python code below
            ```python
            n1, n2 = 1.7, 1.5
            sum = n1 + n2
            print('The sum of {0} and {1} is {2}'.format(n1, n2, sum))
            print('The sum of {} and {} is {}'.format(n1, n2, sum))
            print('The sum of {:.0f} and {:05.2f} is {:5.2f}'.format(n1, n2, sum))
            ```
        - run "ex_sum.py" via **Anaconda Prompt (administrative)**
            ```text
            python c:\PycharmProjects\example\ex_sum.py
            ```
            
1. Install **PyCharm (administrative)**

1. Create "practice" Project based on Conda Environment via **PyCharm (administrative)**

1. Run PyThon Code in **PyCharm (administrative)**
    - create a python file (ex_sum.py)
        ```python
        print('hello world')
        n1, n2 = 1.7, 1.5
        sum = n1 + n2
        print('The sum of {0} and {1} is {2}'.format(n1, n2, sum))
        print('The sum of {} and {} is {}'.format(n1, n2, sum))
        print('The sum of {:.0f} and {:05.2f} is {:5.2f}'.format(n1, n2, sum))
        ```
    - run the python code via **PyCharm (administrative)**
        - Ctrl + Shift + F10
        - Mouse Right Click => Run 'ex_sum.py'
    - exectue selected lines in console via **PyCharm (administrative)**
        - Select codes => Alt + Shift + e
        - Mouse Right Click => Exectue selection in console
    - exectue a line in console via **PyCharm (administrative)**
        - Alt + Shift + e
        - Mouse Right Click => Exectue selection in console)
    - exectue additional line and check variables in console mode via **PyCharm (administrative)**
        ```python
        n1 = 1.2
        square_1 = n1**2
        square_2 = n1*n1
        ```
    
1. Install PyThon Libraries: matplotlib, scipy, h5py
    - install libraries via **Anaconda Prompt (administrative)**
        ```bash
        activate practice
        conda install matplotlib scipy (pip install h5py matplotlib)
        conda uninstall h5py (pip uninstall h5py)        
        ```
    - install libraries via **PyCharm (administrative)**
        - install and remove h5py via Pip Package Manager
        - install h5py via Conda Package Manager
            
    - create and run a python file (example_plot.py)
        ```python
        import torch
        print(torch.__version__)
        print(torch.tensor([[1., -1.], [1., -1.]]))
        ```        
   
    - create and run a python file (example_fft.py)
        ```python
        import torch
        print(torch.__version__)
        print(torch.tensor([[1., -1.], [1., -1.]]))
        ```        
        
1. Create "PyTorch" Project based on a New Conda Environment (tc) via **PyCharm (administrative)**
    - install PyTorch via **Anaconda Prompt (administrative)**
        ```bash
        activate tc
        conda list
        conda install pytorch-cpu torchvision-cpu -c pytorch
        conda list
        ```    
    - install libraries (matplotlib, scify, h5py) via Conda Package Manager in **PyCharm (administrative)**
    - create and run (example_torch.py) via **PyCharm (administrative)**
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
