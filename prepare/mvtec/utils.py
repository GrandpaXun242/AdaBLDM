import os


def get_dataset_categories(datadir, exclude=["good"]):
    categories = os.listdir(datadir)
    for e in exclude:
        try:
            categories.remove(e)
        except:
            continue
    return categories
