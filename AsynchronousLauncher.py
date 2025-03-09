import time
import utils
import os
import libtmux

class RunningCommand:
    def __init__(self, en, command_id, roles, running_command, tmux, run_locally=True):
        self.en = en
        self.command_id = command_id
        self.roles = roles
        self.running_command = running_command
        self.run_locally = run_locally
        self.tmux = tmux

    def wait(self, polling_interval=1):
        """
        Waits for the command to finish.
        """
        while self.is_running():
            print(f"Waiting for command {self.running_command} to finish...")
            time.sleep(polling_interval)

    def kill(self):
        """
        Kills the command.
        """
        if self.run_locally:
            self.en.run_command(f"tmux kill-session -t async_{self.command_id}", roles=self.roles)
        else:
            os.system(f"tmux kill-session -t async_{self.command_id}")

    def is_running(self):
        """
        Returns whether the command is running.
        """
        running_processes = 0
        if self.run_locally:
            for session in self.tmux.sessions:
                if f"async_{self.command_id}" in session.name:
                    return True
            return False
        for role in self.roles:
            try:
                self.en.run_command(f"tmux list-sessions | grep async_{self.command_id}", roles=role)
                running_processes += 1
            except Exception as e:
                pass
        return running_processes > 0



class AsynchronousLauncher:
    def __init__(self, en):
        self.en = en
        self.command_id = 0
        self.tmux = libtmux.Server()

    def run_command(self, command, roles=None, run_locally=True):
        """
        Runs the given command on the given roles.

        Parameters
        ----------
        command : str
            The command to run
        roles : Dict
            The roles of the nodes

        Returns
        -------
        RunningCommand : The running command
        """
        if not run_locally:
            self.en.run_command(f"tmux new-session -d -s async_{self.command_id} '{command}'", roles=roles)
        else:
            self.tmux.cmd('new-session', '-d', '-s', f'async_{self.command_id}', f'{command}')
            
        runningCommand = RunningCommand(self.en, self.command_id, roles, command, self.tmux,run_locally)
        self.command_id += 1
        return runningCommand
    
