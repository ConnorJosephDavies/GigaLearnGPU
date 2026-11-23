import site
import sys
import json
import os

wandb_run = None

# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id = None):

	global wandb_run
	
	# Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path
	
	try:
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		sys.path.append(site_packages_dir)
		site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)
	
	print("Calling wandb.init()...")
	kw = dict(project = project, group = group, name = name)
	if not (id is None) and len(id) > 0:
		kw.update(id = id, resume = "allow")
	try:
		wandb_run = wandb.init(**kw)
	except Exception as e:
		# Never let a network blip kill training. Fall back to offline (sync later with `wandb sync`).
		print(f"wandb online init failed ({e!r}); logging OFFLINE")
		try:
			wandb.finish()
		except Exception:
			pass
		wandb_run = wandb.init(mode = "offline", **kw)
	return wandb_run.id

def add_metrics(metrics):
	global wandb_run
	try:
		wandb_run.log(metrics)
	except Exception as e:
		print(f"wandb log failed: {e!r}")
