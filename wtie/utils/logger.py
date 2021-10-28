import os, subprocess, sys
from pathlib import Path
from datetime import datetime, timedelta

#TODO: look at python logging module



##############################
# LOGGER
##############################

class Logger:
    """Simple logger. Log events to a file."""

    def __init__(self,
                 file_path: str,
                 overwrite: bool=False,
                 ignore_git_info: bool=True,
                 hook_message: bool=False):

        self.file = Path(file_path)

        self.name = self.file.name
        self.directory = self.file.parent.as_posix()

        assert self.file.parent.is_dir(), ("Directory %s should be created first." % self.directory)
        #assert self.file.parent.exists()




        if self.file.exists() and overwrite:
            self.file.unlink()

        if not self.file.exists():
            # create file
            _datetime = str(datetime.now().strftime('%H:%M:%S on %A, %B the %dth, %Y'))
            self.file.write_text(("INITIALIZATION...\nCreation of the log at %s." % _datetime))

            # sys info
            _system = str(os.uname())
            self.write(("System information: %s" % _system))

            # git info
            if ignore_git_info:
                self.write("Git information: ignored")
            else:
                # Can be slow and requires to enter login credentials
                # everytime if user did not exchange ssh keys with the server.
                _git_info = subprocess.getoutput("git remote show origin")
                _git_rev = subprocess.getoutput("git rev-parse HEAD")
                self.write(("Git repository: \n%s" % _git_info))
                self.write(("current git commit revision number: %s\n") % _git_rev)


            # conda info
            _conda_info = subprocess.getoutput("conda info")
            self.write(("Conda information: %s") % _conda_info)

        else:
            _datetime = str(datetime.now().strftime('%H:%M:%S on %A, %B the %dth, %Y'))
            if hook_message:
                self.write(("Hooking logger at %s." % _datetime))


    def write(self, text: str, highlight: bool=False) -> None:
        if highlight:
            text_p = bcolors.BOLD + bcolors.OKCYAN + text + bcolors.ENDC
        else:
            text_p = text
        print(text_p)
        with self.file.open('a') as f:
            f.write('\n' + text)




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


##############################################
# UTILS
##############################################
def print_time_from_seconds(seconds: int):
    sec = timedelta(seconds=int(seconds))
    d = datetime(1,1,1) + sec

    print("DAYS:HOURS:MIN:SEC")
    print("%d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))


def display_time_from_seconds(seconds: int, lower: bool=True):
    sec = timedelta(seconds=int(seconds))
    d = datetime(1,1,1) + sec

    _str = ("DAYS:HOURS:MIN:SEC - %d:%d:%d:%d" % (d.day-1, d.hour, d.minute, d.second))

    if lower: _str = _str.lower()

    return _str


def simple_progressbar(count: int, total: int, refresh_rate: int=1):
    if count % refresh_rate == 0:
        perc = 100*(count/total)
        sys.stdout.write('\r%.1f%% completed' % perc)
        sys.stdout.flush()


def sysout(message: str):
    sys.stdout.write(str(message))
    sys.stdout.flush()