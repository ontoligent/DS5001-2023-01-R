# Mazo

Mazo is a simple inteface to [MALLET](http://mallet.cs.umass.edu/index.php), a state of the art topic modeling tool. Mazo is built on top of [Polite](https://github.com/ontoligent/polite), which is a lite version of [Polo](https://github.com/ontoligent-design/polo2). Mazo is "mallet" in Spanish and is pronounced `/MA-so/`. Mazo uses its own version of Polite, so you don't need to install that separately.

## Installation

First install [MALLET 2.0](https://mimno.github.io/Mallet/). Mazo is a low code wrapper around MALLET, designed to make it easy to generate topic models and to store the resulting outputs in a collection of relational tables (as CSV files). 

Ideally, you will have the path to the `mallet` executable in your environment so that it can be run from anywhere on our system. Or you can create an alias to the executable file in your shell initialization file (e.g. `.bash_profile`). As a final resort, you can give the path in the Mazo config file (see below).

MALLET is sometimes hard to set up, due to its Java dependencies. In some cases you need to make sure your Java classpath includes `lib/mallet.jar`, the `class` directory, and the `lib/mallet-deps.jar`. For example, in `.bash_profile` you would add the following, where `<mallet_home>` is string representing the root path of your MALLET installation:

```
MALLET_HOME=<mallet_home>
CLASSPATH=$MALLET_HOME/lib/mallet.jar:$MALLET_HOME/class:$MALLET_HOME/lib/mallet-deps.jar:$CLASSPATH
```

Then clone this repo and, from within the cloned directory, run `python setup.py install` &mdash; or `python setup.py install --user` if you are on a shared system where you can't write to the python directory. This will install the script `mazo` and the library `polite.polite` into your current Python environment. The script `mazo` will be callable from anywhere on your system.

Note that Mazo requires Python 3.x.

## Usage

Create a working directory for your project and move into it. Create two subdirectories, `./corpus` and `./output` and optionally a configuration file `config.ini`. 

As stated in the installation instructions, Mazo expects `mallet` to be in your `PATH` environment variable. If it is not, you'll need to edit the `config.ini` file. For example, if you are using Windows and followed [the installation instructions](http://mallet.cs.umass.edu/download.php) for MALLET on the website, you'd change the value of `mallet_path` to `bin\mallet`, like so:

```
[DEFAULT]
mallet_path = bin\mallet
``` 

Or, if you want to point to the specific location of `mallet`, you can do something like this:

```
[DEFAULT]
mallet_path = C:\mallet-2.0.8\bin\mallet
```

or on a Unix-based system:

```
[DEFAULT]
mallet_path = /opt/mallet/bin/mallet
```

To begin using Mazo, you'll need to first put [a MALLET compliant corpus file](http://mallet.cs.umass.edu/import.php) in the corpus directory `./corpus` and name it in a special way:

```
<keyword>-corpus.csv
```

Here, `<keyword>` is a word used to name everything else. For example, the corpus directory contains a sample corpus file called `demo-corpus.csv`; `demo` is the keyword. After mazo runs, everything will be put in an output directory with the word `demo` prefixed to the files and directories.

A MALLET compliant corpus file is, in this context, a comma-delimited CSV file with three columns: a document identifier, a label, and the document string. The file should have no header. Note that the document string can have commas; MALLET stops parsing after the second comma. See the `./corpus` for example files.

To run Mazo, do this:

```
mazo <keyword> <k>
```

where `<k>` stands for the number of topics in the model.

To try it out, use the demo file found in the clone repo:

```
mazo demo 20
```

After this runs, in your `./output` directory you will find a directory named someting like this:

```buildoutcfg
output/demo-20-1585231796908834
```
Note, Mazo will create the `./output` if your forgot to.

The long number is just the keyword with the number of tokens and a unix timestamp added to it. It is used to separate your topic models from each other, since each is unique. (You should delete these directories when you done with them.)

Inside of this directory, you will find all the files that MALLET generated, plus a subdirectory `./tables`. In that directory, you should see the following files:

```buildoutcfg
DOC.csv
DOCTOPIC.csv
DOCTOPIC_NARROW.csv
DOCWORD.csv
TOPIC.csv
TOPICPHRASE.csv
TOPICWORD.csv
TOPICWORD_DIAGS.csv
TOPICWORD_NARROW.csv
TOPICWORD_WEIGHTS.csv
VOCAB.csv
```

These files implement a relational data model of the topic model. They can be imported into a relational database (like SQLite) or read directly into Pandas. 

Have fun!

# Other options

To see more options, run `mazo -h`. You will see this:

```
$ mazo -h
usage: mazo [-h] [--iters ITERS] [--trial_key TRIAL_KEY] [--config_file CONFIG_FILE]
            [--print_only PRINT_ONLY] [--save_mode SAVE_MODE]
            keyword n_topics

positional arguments:
  keyword               the corpus keyword
  n_topics              the number of topics

optional arguments:
  -h, --help            show this help message and exit
  --iters ITERS         number of iterations; default = 1000
  --trial_key TRIAL_KEY
                        the name of the trial; default is current timestamp, e.g. 16521990932452369.
  --config_file CONFIG_FILE
                        Mazo config file
  --print_only PRINT_ONLY
                        Only print mallet command config files
  --save_mode SAVE_MODE
                        Save to CSV 'csv' or SQLite 'sql'; default = 'csv'
```

You can change the number or iterations to train the model as well as the trial key, which serves as a suffix to uniquely identify the model data.  

# Troubleshooting

One reason `mazo` will fail is that your corpus has characters that are unreadable by `mallet` when it is importing the corpus file. Strip out high-ASCII characters from the file first. You can use the regular expression `[\040-\176]+` to find the offending characters and replace them with the empty string `''`. The MacOS [BBEdit](https://www.barebones.com/products/bbedit/) has a function to do this called "zap gremlins."

# Final Remarks

Remember that Mazo is meant to get you started using MALLET, and to provide a nice set of output files for downstream analysis, visualization, etc. If you need more power and flexibility, you are encouraged to use MALLET directly. 

If you do use MALLET directly but then want to convert the resulting files into relational tables, consider importing `Polite` from `polite.polite` and using it directly.
