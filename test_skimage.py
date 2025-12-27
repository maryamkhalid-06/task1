import sys
try:
    import skimage
    print(f"skimage version: {skimage.__version__}")
    
    import skimage.measure
    print("skimage.measure imported")
    print("marching_cubes in dir:", 'marching_cubes' in dir(skimage.measure))
    print("marching_cubes_lewiner in dir:", 'marching_cubes_lewiner' in dir(skimage.measure))
    
    from skimage import measure
    print("from skimage import measure worked")
    print("measure has marching_cubes:", hasattr(measure, 'marching_cubes'))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
