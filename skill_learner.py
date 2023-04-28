import numpy as np

def ask_for_skill(skill_name):
    print(f"{skill_name} is being learned")

    def skill(pcd):
        print("The object pcd is at:", np.mean(pcd, axis=0))

    return skill