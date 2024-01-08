CNN Model
=========

Installation
------------

To use this library, Python version 3.8 or later is required.
Refer to https://www.python.org/ for more information.

While not required, the library can easily be installed using Poetry.
For official installation instructions, refer to https://python-poetry.org/.
In short, the following command can be used:

    $ curl -sSL https://install.python-poetry.org | python3 -
    Retrieving Poetry metadata
    
    # Welcome to Poetry!
    
    This will download and install the latest version of Poetry,
    a dependency and package manager for Python.
    
    It will add the `poetry` command to Poetry's bin directory, located at:
    
    /home/ddimitrov/.local/bin
    
    You can uninstall at any time by executing this script with the --uninstall option,
    and these changes will be reverted.
    
    Installing Poetry (1.7.1): Done
    
    Poetry (1.7.1) is installed now. Great!
    
    You can test that everything is set up by executing:
    
    `poetry --version`
    
    $ poetry --version
    Poetry (version 1.7.1)

This repository can be cloned as follows:

    $ git clone git@github.com:dsd65535/cnn_model.git
    Cloning into 'cnn_model'...
    remote: Enumerating objects: 301, done.
    remote: Counting objects: 100% (301/301), done.
    remote: Compressing objects: 100% (106/106), done.
    remote: Total 301 (delta 202), reused 286 (delta 191), pack-reused 0
    Receiving objects: 100% (301/301), 93.87 KiB | 814.00 KiB/s, done.
    Resolving deltas: 100% (202/202), done.
    $ cd cnn_model/

Using Poetry, it can be installed as follows:

    $ poetry install

Usage
-----

Using Poetry, the top-level script can be run as follows:

    $ poetry run cnn_model
