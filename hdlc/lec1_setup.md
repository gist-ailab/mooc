1. Download Setup Files
    - Anaconda :: Windows 64 bit (https://www.anaconda.com/)
    - PyCharm :: Windows Community Version (https://www.jetbrains.com/pycharm/)

1. Install Anaconda in Windows 10 (64-Bit)

1. Create an Environment in Anaconda
     ```text
    conda env list
    conda create --name test (conda create -n test)
    conda env list
    activate test
    deactivate
    conda env remove test
    ```

1. Install PyThon and PyTorch in Anaconda Environment
    ```text
    conda create -n tc
    activate tc
    conda list
    conda install python
    conda list
    conda install pytorch-cpu torchvision-cpu -c pytorch
    deactivate
    ```

1. Run PyThon Code in Anaconda    
    - run python code in console mode
        ```text
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
        ```text
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
1. Install PyCharm

1. Create a Project in PyCharm

1. Run PyThon Code in PyCharm
    - save and run python code (.py)
    - save and run python code (test2.py)
    run python code in console mode
