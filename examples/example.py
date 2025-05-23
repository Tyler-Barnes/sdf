from sdf import *
if __name__ == "__main__":
    f = sphere(1) & box(1.5)
    
    c = cylinder(0.5)
    f -= c.orient(X) | c.orient(Y) | c.orient(Z)
    
    f.save('out.stl')
