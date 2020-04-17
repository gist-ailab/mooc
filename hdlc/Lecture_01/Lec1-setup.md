1. Download Setup Files
    - Miniconda (Windows 64 bit) (https://docs.conda.io/en/latest/miniconda.html)
    - PyCharm Community (Windows) (https://www.jetbrains.com/pycharm/download/#section=windows)

1. Install **Miniconda (for All Users in Administrative Mode)** [about 3 minutes]
    - check `Add Anaconda to the system PATH environment variable`

1. Install **PyCharm** [about 3 minutes]
    - check `Create Desktop Shortcut`, `Create Associations`

1. Set and run **PyCharm (in Administrative Mode)**

1. Create a New Project `example` based on a New Conda Environment `tc`

1. Run Python Code
    - create a python file `ex_sum.py`
        ```python
        print('example of sum')
        n1, n2 = 1.7, 1.5
        sum = n1 + n2
        print('The sum of {0} and {1} is {2}'.format(n1, n2, sum))
        print('The sum of {} and {} is {}'.format(n1, n2, sum))
        print('The sum of {:.0f} and {:05.2f} is {:5.2f}'.format(n1, n2, sum))
        ```
    - run the python code `ex_sum.py`
        - `Mouse Right Click` => Click `Run ex_sum.py`
        - `Ctrl` + `Shift` + `F10`        
    - execute selected lines in console mode
        - `Mouse Right Click` => Click `Execute Selection in Python Console`
        - `Alt` + `Shift` + `E key`        
    - execute a line in console mode
        - `Mouse Right Click` => Click `Execute Line in Python Console`
        - `Alt` + `Shift` + `E key`        
    - execute additional line and check variables in console mode
        ```python
        n1 = 1.2
        square = n1 * n1
        print(square.__str__())
        ```
    - rename variable name
        - `Mouse Right Click` => Click `Refactor` => Click `Rename....`
        - `Shift` + `F6`        
    - reformat the code
        - Click `Code` in menu bar => Click `Reformat Code`
        - `Ctrl` + `Alt` + `L key`
    - switch the code with comments
        - Add `#` in front of the line
        - `Ctrl` + `/ key`
        
1. Install Python Libraries: `matplotlib`, `h5py`, `scipy`    
    - create and run a python file `ex_plot.py`
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
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
    - install libraries: `matplotlib`, `h5py`, `scipy` [about 3 minutes]
    - retry running the python file: `ex_plot.py`
    - create and run a python file: `ex_fft.py`
        ```python
        import matplotlib.pyplot as plt
        import numpy as np
        import os, h5py
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
        
1. Install PyTorch
    - create new project `DeepLearning` based on the Conda Environment `tc`
    - create and run a python file: `ex_torch.py`
        ```python
        import torch
        print(torch.__version__)
        a = torch.tensor([[1., -1.], [1., -1.]])
        print(a)
        print(torch.cuda.is_available())
        if torch.cuda.is_available() == True:
            print(a.cuda())
        ```        
    - check NVIDIA Graphic Card (GPU) to use CUDA
    - add repository url: `pytorch`
    - install library: `torchvision` for NVIDIA GPU users or `torchvision-cpu` for the others [about 5 minutes]
    - retry runnning the python file: `ex_torch.py`
            
1. Tutorials
    - Anaconda Tutorials (https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
    - PyCharm Tutorials (https://www.jetbrains.com/help/pycharm/quick-start-guide.html)
    - PyThon Tutorials (https://docs.python.org/3/tutorial/)
    - Matplotlib Tutorials (https://matplotlib.org/tutorials/index.html)
    - SciPy Tutorials (https://docs.scipy.org/doc/scipy/reference/tutorial/)
    - h5py Tutorials (http://docs.h5py.org/en/stable/quick.html)
    - PyTorch Tutorials (https://pytorch.org/tutorials/)
