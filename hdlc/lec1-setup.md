1. Download Setup Files
    - PyThon: Anaconda Windows 64 bit Version (https://www.anaconda.com/)
    - PyCharm: Windows Community Version (https://www.jetbrains.com/pycharm/)
    - PyTorch: PyTorch 1.1 CPU Version (https://pytorch.org/)


1. Install **Anaconda Prompt (administrative mode)**
    - set Anaconda Prompt (administrative mode) as default    

1. Create an Environment via **Anaconda Prompt**
     ```bash
    conda env list
    conda create --name test
    conda env list
    activate test
    deactivate
    conda env remove -n test
    ```

1. Install PyThon in Anaconda Environment via **Anaconda Prompt**
    ```bash
    conda create -n practice
    activate practice
    conda list
    conda install python
    conda list
    deactivate
    exit
    ```

1. Run PyThon Code via **Anaconda Prompt**
    - run python code in console mode via **Anaconda Prompt**
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
    - save and run python code (sum.py) via **Anaconda Prompt**
        - create a python file (c:\Projects\test\sum.py) containing the python code below
            ```python
            n1, n2 = 1.7, 1.5
            sum = n1 + n2
            print('The sum of {0} and {1} is {2}'.format(n1, n2, sum))
            print('The sum of {} and {} is {}'.format(n1, n2, sum))
            print('The sum of {:.0f} and {:05.2f} is {:5.2f}'.format(n1, n2, sum))
            exit
            exit()
            ```
        - run a python file (ex_sum.py) via **Anaconda Prompt**
            ```bash
            python c:\Projects\test\sum.py
            ```
            
1. Install **PyCharm (administrative mode)**
    - set PyCharm (administrative mode) as default    

1. Create a New Project (example) based on Conda Environment (practice) via **PyCharm**

1. Run PyThon Code in **PyCharm**
    - create a python file (ex_sum.py)
        ```python
        n1, n2 = 1.7, 1.5
        sum = n1 + n2
        print('The sum of {0} and {1} is {2}'.format(n1, n2, sum))
        print('The sum of {} and {} is {}'.format(n1, n2, sum))
        print('The sum of {:.0f} and {:05.2f} is {:5.2f}'.format(n1, n2, sum))
        ```
    - run the python code via **PyCharm**
        - Ctrl + Shift + F10
        - Mouse Right Click => Run 'ex_sum.py'
    - exectue selected lines in console via **PyCharm**
        - Alt + Shift + e
        - Mouse Right Click => Click 'Exectue Selection in Console'
    - exectue a line in console via **PyCharm**
        - Alt + Shift + e
        - Mouse Right Click => Click 'Exectue Line in Console'
    - exectue additional line and check variables in console mode via **PyCharm**
        ```python
        n1 = 1.2
        square_1 = n1*n1
        square_2 = n1**2
        ```
    
1. Install PyThon Libraries: matplotlib, scipy, h5py
    - install libraries via **Anaconda Prompt**
        ```bash
        activate practice
        conda install matplotlib scipy
        pip install h5py
        pip uninstall h5py
        conda uninstall h5py
        ```
    - install libraries via **PyCharm**
        - install and remove h5py via Pip Package Manager
        - install h5py via Conda Package Manager
            
    - create and run a python file (ex_plot.py)
        ```python
        import numpy as np
        import matplotlib.pyplot as plt
        import h5py
            
        f = [3, 5]  # Hz
        f_sampling = 50
        t = np.linspace(0, 2, 2 * f_sampling)
        a = 0.7 * np.sin(2 * np.pi * f[0] * t) + 0.5 * np.sin(2 * np.pi * f[1] * t)

        data_file = 'data.h5'
        with h5py.File(data_file, 'w') as f:
            f.create_dataset('f_sampling', data=f_sampling)
            f.create_dataset('t', data=t)
            f.create_dataset('a', data=a)
            
        figure, axis = plt.subplots()
        axis.plot(t, a)
        axis.set_title('Signal')
        axis.set_xlabel('Time [s]')
        axis.set_ylabel('Amplitude')
        axis.grid()
        plt.show()
        ```        
   
    - create and run a python file (ex_fft.py)
        ```python
        import os, h5py
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import fftpack

        data_file = 'data.h5'
        if os.path.exists(data_file):
            with h5py.File(data_file, 'r') as f:
                print(f.keys())
                f_sampling = f['f_sampling'][()]
                t, a = f['t'][()], f['a'][()]
                
        m = fftpack.fft(a) / len(a) * 2
        frequency = fftpack.fftfreq(len(a)) * f_sampling
            
        figure, axis = plt.subplots()
        axis.stem(frequency, np.abs(m))
        axis.set_title('Frequency Spectrum')
        axis.set_xlabel('Frequency [Hz]')
        axis.set_ylabel('Magnitude')
        axis.set_xlim(0, 8)
        axis.set_ylim(0, 1)
        axis.grid()
        plt.show()
        ```        
        
1. Create a New Project (PyTorch) based on a New Conda Environment (tc) via **PyCharm**
    - install matplotlib, scipy, h5py via **PyCharm**    
    - install PyThon, PyTorch via **Anaconda Prompt**
        ```bash
        activate tc
        conda install pytorch-cpu torchvision-cpu -c pytorch
        ```    
    - create and run (ex_torch.py) via **PyCharm**
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
