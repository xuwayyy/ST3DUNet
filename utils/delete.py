import os
import shutil

def clean_empty_ckpt_dirs(ckpt_root, exts=(".pth",)):
    subdirs = [
        os.path.join(ckpt_root, d)
        for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d))
    ]

    if not subdirs:
        print("⚠️ No checkpoint directories found.")
        return

    subdirs_sorted = sorted(subdirs, key=os.path.getctime)
    latest_dir = subdirs_sorted[-1]

    deleted = 0
    for subdir in subdirs_sorted:
        if subdir == latest_dir:
            continue  # save latest checkpoint dir

        has_ckpt = any(
            fname.endswith(ext)
            for fname in os.listdir(subdir)
            for ext in exts
        )
        if not has_ckpt:
            shutil.rmtree(subdir)
            print(f"🗑️  Deleted empty checkpoint dir: {subdir}")
            deleted += 1

    if deleted == 0:
        print("✅ No empty checkpoint directories found (except latest one).")
    else:
        print(f"🧹 Cleaned {deleted} empty checkpoint directories (latest kept: {latest_dir}).")


def clean_empty_visual_dirs(vis_root):
    subdirs = [
        os.path.join(vis_root, d)
        for d in os.listdir(vis_root)
        if os.path.isdir(os.path.join(vis_root, d))
    ]

    if not subdirs:
        print("⚠️ No visualization directories found.")
        return

    subdirs_sorted = sorted(subdirs, key=os.path.getctime)
    latest_dir = subdirs_sorted[-1]

    deleted = 0
    for subdir in subdirs_sorted:
        if subdir == latest_dir:
            continue  # the latest visualization dir is kept

        if not has_any_file_recursive(subdir):
            shutil.rmtree(subdir)
            print(f"🗑️  Deleted empty visualization dir: {subdir}")
            deleted += 1

    if deleted == 0:
        print("✅ No empty visualization directories found (except latest one).")
    else:
        print(f"🧹 Cleaned {deleted} empty visualization directories (latest kept: {latest_dir}).")



def has_any_file_recursive(dir_path):
    for root, dirs, files in os.walk(dir_path):
        if files:  
            return True
    return False

def clean_orphan_log_dirs(log_root, ckpt_root, vis_root):
    log_subdirs = [
        os.path.join(log_root, d)
        for d in os.listdir(log_root)
        if os.path.isdir(os.path.join(log_root, d))
    ]

    ckpt_subdirs_names = {
        d for d in os.listdir(ckpt_root)
        if os.path.isdir(os.path.join(ckpt_root, d))
    }

    vis_subdirs_names = {
        d for d in os.listdir(vis_root)
        if os.path.isdir(os.path.join(vis_root, d))
    }

    if not log_subdirs:
        print("⚠️ No log directories found.")
        return

    log_subdirs_sorted = sorted(log_subdirs, key=os.path.getctime)
    latest_log_dir = log_subdirs_sorted[-1]

    deleted = 0
    for log_dir in log_subdirs_sorted:
        log_dir_name = os.path.basename(log_dir)
        if log_dir == latest_log_dir:
            continue  

        if log_dir_name not in ckpt_subdirs_names and log_dir_name not in vis_subdirs_names:
            shutil.rmtree(log_dir)
            print(f"🗑️  Deleted orphan log dir: {log_dir}")
            deleted += 1

    if deleted == 0:
        print("✅ No orphan log directories found (except latest one).")
    else:
        print(f"🧹 Cleaned {deleted} orphan log directories (latest kept: {latest_log_dir}).")