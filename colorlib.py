import numpy as np

#################################################
#####  Constants used in cieLUV conversion.  ####
#################################################
EPSILON = 216.0 / 24389.0
KAPPA   = 24389.0 / 27.0
#################################################

#################################################################
#################################################################
M = np.array(                                                   \
[                                                               \
    [3.240969941904521,-1.537383177570093,-0.498610760293283],  \
    [-0.96924363628087, 1.875967501507720, 0.041555057407175],  \
    [0.055630079696993,-0.203976958888970, 1.056971514242878],  \
]                                                               )
#################################################################

#############################################################################
#########################  2° reference U and V  ############################
###############  calculated from the respective xyz values  #################
########################  [95.047,100.0,108.883].  ##########################
#############################################################################
REF_UV = np.asarray([ 0.19783000664283680764, 0.46831999493879100370])
#############################################################################

#############################################################################
##################  Transformation matrices between RGB  ####################
#################  and XYZ based on a 2° reference and D65  #################
#################  color temperature for the white point.  ##################
#############################################################################
RGB2XYZ = np.asarray(                                                       \
[                                                                           \
[ 0.41239079926595948129, 0.35758433938387796373, 0.18048078840183428751],  \
[ 0.21263900587151035754, 0.71516867876775592746, 0.07219231536073371500],  \
[ 0.01933081871559185069, 0.11919477979462598791, 0.95053215224966058086],  \
]                                                                           )

XYZ2RGB = np.asarray(                                                       \
[                                                                           \
[ 3.24096994190452134377, -1.53738317757009345794,-0.49861076029300328366], \
[-0.96924363628087982613,  1.87596750150772066772, 0.04155505740717561247], \
[ 0.05563007969699360846, -0.20397695888897656435, 1.05697151424287856072], \
]                                                                           )
#############################################################################

#################################################
########  RGB <-> bt.601 YUV matrices.  #########
#################################################
RGBB601 = np.asarray(                           \
[                                               \
        [ 0.29900,    0.58700,   0.11400],      \
        [-0.14713,   -0.28886,   0.43600],      \
        [ 0.61500,   -0.51499,  -0.10001],      \
]                                               )

B601RGB = np.asarray(                           \
[                                               \
        [1.00000,    0.00000,    1.13983],      \
        [1.00000,   -0.39465,   -0.58060],      \
        [1.00000,    2.03211,    0.00000],      \
]                                               )
#################################################

def clipRGB(img):
    [ab,ag,ar] = as_channels(img-127.5)
    m = np.maximum(np.abs(ab),np.maximum(np.abs(ag),np.abs(ar)))
    m = as_pixels(np.asarray([m,m,m]))
    img = as_pixels([ab,ag,ar])
    c = m > 127.5
    img[c] = 127.5*img[c]/m[c]
    return img + 127.5

def make_LUT(nplut,ls,fname):
    nplut = np.flip(nplut,2)/255.0
    nplut = nplut[0]
    nplut[np.isnan(nplut)] = 0.0
    with open(fname+".cube", 'w') as lut_file:
        lut_file.write('LUT_3D_SIZE '+str(ls)+'\n\n')
        for i in range(ls**3):
            s = ("%.8f %.8f %.8f" % tuple(nplut[i]))
            lut_file.write("%s\n" % s)

#####################################################################
##################  I M A G E   A S   P I X E L S :  ################
##############  N-D array (pixel positions) containing  #############
#######  3-D arrays (channel values of each pixel) as elements.  ####
#####################################################################
################# I M A G E   A S   C H A N N E L S : ###############
##################  3-D array (channels) containing  ################
####  N-D arrays (position arrays of each channel) as elements.  ####
#####################################################################
def as_channels(image):
    # From image as pixels to image as channels.
    return np.moveaxis(image,-1,0)

def as_pixels(image):
    # From image as channels to image as pixels.
    return np.moveaxis(image,0,-1)

def pixel_product(image,matrix):
    # Takes image as pixels, computes matrix*pixel for each of the
    # pixels and returns the new image as channels.
    return np.asarray([np.sum(matrix[i]*image,-1) for i in range(3)])
#####################################################################


#############################################################
#######  C O L O R S P A C E   C O N V E R S I O N S  #######
#############################################################

def bgr2xyz(bgr):
    rgb = np.flip(bgr,2)/255.0
    large = rgb > 0.04045
    small = np.logical_not(large)
    rgb[large] = 100.0*np.power((rgb[large]+0.055)/1.055,2.4)
    rgb[small] = 100.0*rgb[small]/12.92
    xyz = pixel_product(rgb,RGB2XYZ)
    return as_pixels(xyz)

def xyz2bgr(xyz):
    rgb = pixel_product(xyz/100,XYZ2RGB)
    large = rgb>0.0031308
    small = np.logical_not(large)
    rgb[large] = 1.055*np.power(rgb[large],1/2.4)-0.055
    rgb[small] = 12.92*rgb[small]
    bgr = 255*np.clip(np.flip(as_pixels(rgb),2),-200.0,200.0)
    return bgr

def xyz2yxy(xyz):
    [X,Y,Z] = as_channels(xyz)
    x = X/(X+Y+Z)
    y = Y/(X+Y+Z)
    return as_pixels([Y,x,y])

def yxy2xyz(yxy):
    [Y,x,y] = as_channels(yxy)
    yy = Y/y
    X = x*yy
    Z = (1 - x - y)*yy
    return as_pixels([X,Y,Z])

def bgr2yxy(bgr):
    return xyz2yxy(bgr2xyz(bgr))

def yxy2bgr(yxy):
    return xyz2bgr(yxy2xyz(yxy))

def xyz2luv(xyz):
    xyz = as_channels(xyz)
    den = (xyz[0]+15*xyz[1]+3*xyz[2])
    var_uv = np.asarray([4*xyz[0]/den, 9*xyz[1]/den])
    var_uv[np.isnan(var_uv)] = 0
    var_y = xyz[1]/100.0
    large = var_y > EPSILON
    small = np.logical_not(large)
    var_y[large] = 116*np.power(var_y[large],1/3) - 16
    var_y[small] = KAPPA*var_y[small]
    l = var_y
    uv = [13*l*(var_uv[i]-REF_UV[i]) for i in range(2)]
    return as_pixels(np.asarray([l,uv[0],uv[1]]))

def luv2xyz(luv):
    [l,u,v] = as_channels(luv)
    var_y = (l+16)/116.0
    y3 = np.power(var_y,3)
    large = l > KAPPA*EPSILON
    small = np.logical_not(large)
    var_y[large] = y3[large]
    var_y[small] = l[small]/KAPPA
    var_u = u/(13*l) + REF_UV[0]
    var_v = v/(13*l) + REF_UV[1]
    y = 100*var_y
    x = -9*y*var_u/((var_u-4)*var_v-var_u*var_v)
    z = (9*y-(15*var_v*y)-(var_v*x))/(3*var_v)
    x[np.isnan(x)] = 0
    z[np.isnan(z)] = 0
    return as_pixels([x,y,z])

def bgr2luv(bgr):
    return xyz2luv(bgr2xyz(bgr))

def luv2bgr(luv):
    return xyz2bgr(luv2xyz(luv))

def luv2lch(luv):
    luv = as_channels(luv)
    l,uv = luv[0],luv[1:]
    h = np.degrees(np.arctan2(uv[1],uv[0]))
    c = np.sqrt(uv[0]*uv[0]+uv[1]*uv[1])
    s = (1/4.3)*c/l
    s[l==0] = 0
    return as_pixels(np.asarray([h,c,l]))

def lch2luv(hsl):
    [h,s,l] = as_channels(hsl)
    c = s
    h = np.radians(h)
    u = c*np.cos(h)
    v = c*np.sin(h)
    u[np.isnan(h)] = 0
    v[np.isnan(h)] = 0
    return as_pixels([l,u,v])

def bgr2lch(bgr):
    luv = bgr2luv(bgr)
    return luv2lch(luv)

def lch2bgr(hsl):
    luv = lch2luv(hsl)
    return luv2bgr(luv)

def ray_length_until_intersect(hrad, bound):
    return bound[1]/(np.sin(hrad) - bound[0]*np.cos(hrad))

def get_bounds(l):
    result = []
    sub = np.power((l + 16),3)/1560896
    cond = sub <= EPSILON
    sub[cond] = 1/KAPPA
    g = 0
    while g < 3:
        c = g
        g = g + 1
        m = M[c]
        g1 = 0
        while g1 < 2:
            t = g1
            g1 = g1 + 1
            top1 = m@[284517,0,-94839]*sub
            top2 = (m@[731718,769860,838422]*sub - 769860*t)*l
            bottom = m@[0,-126452,632260]*sub + 126452*t
            result.append([top1/bottom, top2/bottom])
    return result

def max_chroma_for_lh(l,h):
    hrad = np.radians(h)
    bounds = get_bounds(l)
    length = np.asarray([b[1]/(np.sin(hrad)-b[0]*np.cos(hrad)) for b in bounds])
    cond = np.isnan(hrad)
    length[length<=0] = np.inf
    return np.min(length,0)

def lch2hsluv(lch):
    [h,c,l] = as_channels(lch)
    cond1 = l > 100-1e-7
    cond2 = l < 1e-8
    cond = np.logical_or(cond1, cond2)
    c[cond] = 0
    l[cond1] = 100
    l[cond2] = 0
    maxc = max_chroma_for_lh(l,h)
    s = c*(100/maxc)
    return as_pixels([h,s,l])

def hsluv2lch(hsl):
    [h,s,l] = as_channels(hsl)
    cond1 = l > 100-1e-7
    cond2 = l < 1e-8
    cond = np.logical_or(cond1, cond2)
    s[cond] = 0
    l[cond1] = 100
    l[cond2] = 0
    maxc = max_chroma_for_lh(l,h)
    c = s*(maxc/100)
    return as_pixels([h,c,l])

def luv2hsluv(luv):
    return lch2hsluv(luv2lch(luv))

def hsluv2luv(hsl):
    return lch2luv(hsluv2lch(hsl))

def bgr2hsluv(bgr):
    return lch2hsluv(bgr2lch(bgr))

def hsluv2bgr(hsl):
    return lch2bgr(hsluv2lch(hsl))

def hueAux(h,n):
    k = (n + h*6.0) % 6
    return np.clip(np.minimum(k,4-k),0.0,1.0)

def hsv2bgr(hsv):
    hsv = as_channels(hsv)
    gray = np.isnan(hsv[0])
    vs = hsv[1]*hsv[2]
    r = hsv[2]-vs*hueAux(hsv[0],5.0)
    g = hsv[2]-vs*hueAux(hsv[0],3.0)
    b = hsv[2]-vs*hueAux(hsv[0],1.0)
    r[gray] = 1.0*hsv[2][gray]
    g[gray] = 1.0*hsv[2][gray]
    b[gray] = 1.0*hsv[2][gray]
    rgb = np.zeros(np.shape(hsv))
    rgb[2] = r
    rgb[1] = g
    rgb[0] = b
    return 255*as_pixels(rgb)

def bgr2bt601(bgr):
    [b,g,r] = as_channels(bgr)/255.0
    y =  0.29900*r + 0.58700*g + 0.11400*b
    u = -0.14713*r - 0.28886*g + 0.43600*b
    v =  0.61500*r - 0.51499*g - 0.10001*b
    return as_pixels([y,u,v])

def bt6012bgr(yuv):
    [y,u,v] = as_channels(yuv)
    r =  y + 0.00000*u + 1.13983*v
    g =  y - 0.39465*u - 0.58060*v
    b =  y + 2.03211*u + 0.00000*v
    return as_pixels([b,g,r])*255.0

def bgr2yuv(bgr):
    [b,g,r] = as_channels(bgr)/255.0
    y =  (1/3)*r + (1/3)*g + (1/3)*b
    s = np.sqrt(2)/2
    u =  0.0*r - s*g + s*b
    v =  1.0*r - 0.5*g - 0.5*b
    return as_pixels([y,u,v])

def yuv2bgr(yuv):
    [y,u,v] = as_channels(yuv)
    s = np.sqrt(2)/2
    r =  y + (2/3)*v
    g =  y - s*u - (1/3)*v
    b =  y + s*u - (1/3)*v
    return as_pixels([b,g,r])*255.0

def hsl2bgr(hsl):
    [h,s,l] = as_channels(hsl)
    h = np.radians(h)
    u = -s*np.sin(h)/100
    v =  s*np.cos(h)/100
    y = l/100
    yuv = as_pixels([y,u,v])
    return yuv2bgr(yuv)

def bgr2hsl(bgr):
    [y,u,v] = as_channels(bgr2yuv(bgr))
    h =  np.degrees(np.arctan2(-u,v))
    neg = h < 0
    h[neg] = h[neg] + 360
    s = 100*np.sqrt(np.power(u,2)+np.power(v,2))
    l = 100*y
    return as_pixels([h,s,l])

def hsl2luv(hsl):
    return bgr2luv(hsl2bgr(hsl))

def luv2hsl(lsh):
    return bgr2hsl(luv2bgr(lsh))

def hsl2lch(hsl):
    return bgr2lch(hsl2bgr(hsl))

def lch2hsl(lsh):
    return bgr2hsl(lch2bgr(lsh))

def hsl2hsluv(hsl):
    return lch2hsluv(hsl2lch(hsl))

def hsl2hsluv(hsl):
    return lch2hsl(hsluv2lch(hsl))

#############################################################


######################################################################
############################  L E R P S  #############################
######################################################################
def arrlerp(x,x1,y1,x2,y2):
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a*x+b

def p_arrlerp(x,x1,y1,x2,y2):
    x = (x - x1) % (x2 - x1) + x1
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a*x+b

def lerp_trunc(x1,y1,x2,y2):
    if x1 == x2: return lambda z : 0
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return lambda z : (a*z + b) * np.logical_and(z >= x1, z < x2)

def lin_interp(xy):
    [x,y] = xy[xy[:,0].argsort()].transpose()
    l = len(x) - 1
    x0, xf = x[0], x[l]
    b0, bf = y[0] - x0, y[l] - xf
    f = [ lambda z : (z + b0) * (z < x0) ]
    f = f + [ lerp_trunc(x[i],y[i],x[i+1],y[i+1]) for i in range(l) ]
    f = f + [ lambda z: (z + bf) * (z >= xf) ]
    return lambda z: sum(f[i](z) for i in range(len(f)))

def rmod(x,mod):
    m = ((np.array([x])+mod)%(2*mod))-mod
    m[m==-mod/2] = mod/2
    return m[0]

def per_interp(hue, scale):
    first = hue[0,0]
    hue = ((hue)/scale) % 1.0
    l = len(hue)
    hue = hue[hue[:,0].argsort()]
    fl = np.array([hue[l-1],hue[0]])
    fl = fl + np.array([[-1,-1],[1,1]])
    hue = np.append([fl[0]],hue,0)
    [x,y] = np.append(hue,[fl[1]],0).transpose()
    [x,y] = rmod([x,y],2)
    for i in range(l+1):
        yy = np.array([y[i+1]-1,y[i+1],y[i+1]+1])
        s = np.abs((yy-y[i])/(x[i+1]-x[i]))
        if np.isnan(s).any():
            sn=0
        else:
            sm = s[s<np.max(s)]
            if len(sm) <= 1:
                sn = 0
            elif sm[0] == sm[1]:
                sn = 0
            else:
                sn = np.where(s==np.min(sm))[0] - 1
        y[i+1:] = y[i+1:] + sn
    f = [ lerp_trunc(x[i],y[i],x[i+1],y[i+1]) for i in range(1,l+1) ]
#    import matplotlib.pyplot as plt
#    g = lambda z: scale*(sum(f[i](z/scale) for i in range(len(f))) % 1.0)
#    dom = np.linspace(0,360,num=720)
#    plt.scatter(dom,g(dom),s=8)
#    plt.show()
    return lambda z: scale*(sum(f[i](z/scale) for i in range(len(f))) % 1.0)

######################################################################

def linrange(x, interval):
    return np.logical_and(x > interval[0], x < interval[1])

def radrange(x, interval):
    a, b = interval[0]%1, interval[1]%1
    if a >= b:
        cond = np.logical_or(x >= a, x < b)
    else:
        cond = np.logical_and(x > a, x < b)
    return cond

def truncate(img, interval, smooth, scale):
    s = smooth/scale
    i = np.asarray(interval) / scale
    img = img/scale
    if scale == 360:
        scale = scale*3.6
        hshifts = ((np.array([img-i[0]-s/2,img+i[1]+s/2])) % 1.0 + 0.5) % 1.0 - 0.5
        w = 1.0*(radrange(img, i))
        if s > 0:
            ss = np.asarray([-s,s])/2
            b_cond = np.array([\
                    radrange(img,ss+i[0]-s/2),\
                    radrange(img,ss+i[1]+s/2)])
            w[b_cond[0]] = np.maximum(w[b_cond[0]],p_arrlerp(img[b_cond[0]],i[0]-s,0,i[0]+0,1))
            w[b_cond[1]] = np.maximum(w[b_cond[1]],p_arrlerp(img[b_cond[1]],i[1]-0,1,i[1]+s,0))
    else:
        border = np.asarray([[-s,0],[0,s]])
        w = 1.0*(linrange(img, i))
        if s > 0:
            b = (border.transpose() + i).transpose()
            b_cond = [linrange(img,b[0]), linrange(img,b[1])]
            ss = np.asarray([-s,s])/2
            b_cond = np.array([\
                    linrange(img,ss+i[0]-s/2),\
                    linrange(img,ss+i[1]+s/2)])
            w[b_cond[0]] = np.maximum(w[b_cond[0]],arrlerp(img[b_cond[0]],i[0]-s,0,i[0],1))
            w[b_cond[1]] = np.maximum(w[b_cond[1]],arrlerp(img[b_cond[1]],i[1],1,i[1]+s,0))
    return w

def restrict_func(image, interval, smooth, ch_type = 0):
    # 'ch_type' is the type of the channel
    # 0 if it's an RGB channel
    # 1 if it's cieLUV or chroma from cieLCh
    # 2 if it's hue from cieLCh
    img = as_channels(image)
    w = []
    if ch_type == 0:
        for i in range(3):
            if interval[i] == [0,255]:
                w.append(np.isnan(img[1])*0+1)
            else:
                w.append(truncate(img[i],interval[i],smooth*2.55,1.0))
                w[i][np.isnan(w[i])] = 0
    elif ch_type == 1:
        for i in range(3):
            if interval[i] == [0,100]:
                w.append(np.isnan(img[1])*0+1)
            else:
                w.append(truncate(img[i],interval[i],smooth,1.0))
                w[i][np.isnan(w[i])] = 0
    elif ch_type == 2:
        if (interval[0][1] - interval[0][0]) % 360.0 < 0.002:
            if interval[0][0] < interval[0][1]:
                w.append(np.isnan(img[1])*0+1)
            else:
                w.append(np.isnan(img[1])*0)
        else:
            w.append(truncate(img[0],interval[0],smooth,360.0))
        for i in range(1,3):
            if interval[i] == [0,100]:
                w.append(np.isnan(img[1])*0+1)
            else:
                w.append(truncate(img[i],interval[i],smooth,100.0))
                ws = w[i][np.logical_and(w[i] > 0, w[i] < 1)]
    else:
        raise ValueError("'ch_type' must be either 0, 1, or 2.")
    return w[0]*w[1]*w[2]

def restrict(inp, smooth,r_range, g_range, b_range,\
            c_range, h_range, l_range,u_range, v_range):
    clip = []
    [ clip.append(i[0] > 0 or i[1] < 255) for i in [b_range,g_range,r_range] ]
    [ clip.append(i[0] > 0 or i[1] < 100) for i in [l_range,u_range,v_range] ]
    clip.append(h_range[0] % 360 != h_range[1] % 360)
    clip.append(c_range[0] > 0 or c_range[1] < 100)
    weights = inp*0.0 + 1.0
    if any(clip[0:3]):
        weight = restrict_func(inp, [b_range,g_range,r_range], smooth, 0)
        weights = weights*as_pixels([weight,weight,weight])
    if any(clip[3:6]):
        inp = bgr2luv(inp)
        weight = restrict_func(inp, [l_range,u_range,v_range], smooth, 1)
        weights = weights*as_pixels([weight,weight,weight])
        if any(clip[6:8]):
            inp = luv2hsl(inp)
            weight = restrict_func(inp, [h_range,c_range,l_range], smooth, 2)
            weights = weights*as_pixels([weight,weight,weight])
    elif any(clip[6:8]):
            inp = bgr2hsl(inp)
            weight = restrict_func(inp, [h_range,c_range,l_range], smooth, 2)
            weights = weights*as_pixels([weight,weight,weight])
    return weights

def select_channel(inp, channel):
    perp = False
    if   channel == 'blue':  ind = 0
    elif channel == 'green': ind = 1
    elif channel == 'red':   ind = 2
    else:
        cs = 0
        out = bgr2yxy(inp.copy())
        if channel == 'luminance': ind = 0
        else:
            cs = 1
            out = bgr2luv(inp.copy())
            if   channel == 'lightness': ind = 0
            elif channel == 'chroma_u':  ind = 1
            elif channel == 'chroma_v':  ind = 2
            else:
                cs = 2
                out = bgr2hsl(inp.copy())
                if channel == 'hue':
                    ind = 0
                    perp = True
                elif channel == 'chroma':
                    ind = 1
                else:
                    raise ValueError("Invalid channel! The variable 'channel' must be either 'red', 'green', 'blue', 'luminance', 'lightness', 'hue', 'chroma', 'chroma_u', or 'chroma_v'")
    return [out, cs, ind, perp]

def _curve1d(src, curve = [0,0,360,360], channel = 'luminance', smooth = 5,\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], relative_chroma = False):
    inp = src.copy()
    cs = -1
    try:
        points = np.array(curve).reshape(int(len(curve)/2),2)
    except ValueError:
        print("Error: 'curve' list must be of even size!")

    [inp, cs, ind, perp] = select_channel(inp, channel)
    out = inp.copy()
    if perp:
        f = per_interp(points,360)
    else:
        f = lin_interp(points)
    out[:,:,ind] = f(out[:,:,ind])

    if cs == 2:
        if relative_chroma:
            out = hsl2hsluv(out)
            inp = bgr2hsluv(src)
            out[:,:,2] = inp[:,:,2]
            if perp:
                out[:,:,1] = inp[:,:,1]
            out = hsluv2bgr(out)
            inp = hsluv2bgr(inp)
        else:
            out = hsl2lch(out)
            inp = bgr2lch(src)
            out[:,:,2] = inp[:,:,2]
            out = lch2hsluv(out)
            out[:,:,1] = np.clip(out[:,:,1],0.0,100.0)
            out = hsluv2bgr(out)
            inp = lch2bgr(inp)
    elif cs == 1:
        out = luv2hsluv(out)
        if relative_chroma:
            inp = luv2hsluv(inp)
            out[:,:,1] = inp[:,:,1]
            inp = hsluv2bgr(inp)
        else:
            out[:,:,1] = np.clip(out[:,:,1],0.0,100.0)
            inp = luv2bgr(inp)
        out = hsluv2bgr(out)
    elif cs == 0:
        out = yxy2bgr(out)
    weights = restrict(inp, smooth,r_range, g_range, b_range,\
                c_range, h_range, l_range,u_range, v_range)
    out = inp + weights*(out.copy() - inp)
    return out

def _rgbcurve(inp, red=[0,0,255,255], green=[0,0,255,255], blue=[0,0,255,255],\
            mode = None, smooth = 0, gimpfile = None,\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], relative_chroma=False):
    if gimpfile != None:
        file = open(gimpfile, "r")
        lines = file.readlines()
        red = np.array(lines[2].split()).astype(float)
        red = red[red!=-1]
        green = np.array(lines[3].split()).astype(float)
        green = green[green!=-1]
        blue = np.array(lines[4].split()).astype(float)
        blue = blue[blue!=-1]
    try:
        red = np.array(red).reshape(int(len(red)/2),2)
        r = lin_interp(red)
    except ValueError:
        print("Error: 'red' list must be of even size!")
    try:
        green = np.array(green).reshape(int(len(green)/2),2)
        g = lin_interp(green)
    except ValueError:
        print("Error: 'green' list must be of even size!")
    try:
        blue = np.array(blue).reshape(int(len(blue)/2),2)
        b = lin_interp(blue)
    except ValueError:
        print("Error: 'blue' list must be of even size!")
    out = as_channels(inp).copy()
    out[2] = r(out[2])
    out[1] = g(out[1])
    out[0] = b(out[0])
    out = as_pixels(out).clip(0.0,255.0)
    weights = restrict(inp, smooth,r_range, g_range, b_range,\
                c_range, h_range, l_range,u_range, v_range)
    out = inp + weights*(out.copy() - inp)
    if mode == 'chroma':
        inp = as_channels(bgr2lch(inp))
        out = as_channels(bgr2lch(out))
        out[0] = inp[0]
        out[2] = inp[2]
        out = lch2bgr(as_pixels(out))
    if mode == 'color':
        inp = as_channels(bgr2luv(inp))
        out = as_channels(bgr2luv(out))
        out[0] = inp[0]
        out = luv2bgr(as_pixels(out))
    if mode == 'lightness':
        inp = as_channels(bgr2luv(inp))
        out = as_channels(bgr2luv(out))
        if relative_chroma:
            inp = as_channels(as_pixels(luv2hsluv(inp)))
            out = as_channels(as_pixels(luv2hsluv(out)))
            out[0] = inp[0].copy()
            out[1] = inp[1].copy()
        else:
            out[1] = inp[1]
            out[2] = inp[2]
            out = as_channels(luv2hsluv(as_pixels(out)))
            inp = as_channels(luv2hsluv(as_pixels(inp)))
        out[1] = np.clip(out[1],0.0,100.0)
        out = hsluv2bgr(as_pixels(out))
    if mode == 'luminance':
        inp = as_channels(bgr2yxy(inp.copy()))
        out = as_channels(bgr2yxy(out.copy()))
        out[1] = inp[1]
        out[2] = inp[2]
        out = yxy2bgr(as_pixels(out))
    if mode == 'chrominance':
        inp = as_channels(bgr2yxy(inp.copy()))
        out = as_channels(bgr2yxy(out.copy()))
        out[0] = inp[0]
        out = yxy2bgr(as_pixels(out))
    return out


def _perturb(src,x=[0,100],y=[0,0],domain='lightness',codomain='lightness',\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], smooth=5,\
            relative_chroma = False, mode = 'additive'):
    try:
        points = as_pixels([x,y])
    except ValueError:
        print("Error: 'domain' and 'codomain' lists are not the same size!")

    [do, cs1, ind1, perp1] = select_channel(src, domain)
    [co, cs2, ind2, perp2] = select_channel(src, codomain)
    inp = co.copy()

    if perp1:
        f = per_interp(points, 360)
    else:
        a = np.array([100,0])
        l = len(points)
        points = np.append([points[0]-a], points,0)
        points = np.append(points, [points[l]+a],0)
        f = lin_interp(points)
    perturbation = f(do[:,:,ind1])
    if mode == 'additive':
        co[:,:,ind2] = co[:,:,ind2] + perturbation
    elif mode == 'multiplicative':
        co[:,:,ind2] = (1+co[:,:,ind2])*perturbation
    else:
        raise ValueError("'mode' must be either 'additive' or 'multiplicative'")

    [out, cs, ind, perp] = [co, cs2, ind2, perp2]

    if cs == 2:
        if relative_chroma:
            out = hsl2hsluv(out)
            inp = bgr2hsluv(src)
            out[:,:,2] = inp[:,:,2]
            if perp:
                out[:,:,1] = inp[:,:,1]
            out = hsluv2bgr(out)
            inp = hsluv2bgr(inp)
        else:
            out = hsl2lch(out)
            inp = bgr2lch(src)
            out[:,:,2] = inp[:,:,2]
            out = lch2hsluv(out)
            out[:,:,1] = np.clip(out[:,:,1],0.0,100.0)
            out = hsluv2bgr(out)
            inp = lch2bgr(inp)
    elif cs == 1:
        out = luv2hsluv(out)
        if relative_chroma:
            inp = luv2hsluv(inp)
            out[:,:,1] = inp[:,:,1]
            inp = hsluv2bgr(inp)
        else:
            out[:,:,1] = np.clip(out[:,:,1],0.0,100.0)
            inp = luv2bgr(inp)
        out = hsluv2bgr(out)
    elif cs == 0:
        out = yxy2bgr(out)
    weights = restrict(inp, smooth,r_range, g_range, b_range,\
                c_range, h_range, l_range,u_range, v_range)
    out = inp + weights*(out.copy() - inp)
    return out

def _balance(inp, grays = [[0,0,0],[255,255,255]]):
    tinted = bgr2luv([np.flip(grays,1)])[0]
    tinted = tinted[tinted[:,0].argsort()]
    domain = tinted[:,0]
    pert_u = -tinted[:,1]
    pert_v = -tinted[:,2]
    out = inp.copy()
    out = _perturb(out,x=domain,y=pert_u,domain='lightness',codomain='chroma_u')
    out = _perturb(out,x=domain,y=pert_v,domain='lightness',codomain='chroma_v')
    return out

def _tweak(inp, hue=0.0, chroma=1.0, bright=0,\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], smooth=5, relative_chroma=False):
    out = inp.copy()
    if hue != 0.0:
        out = _perturb(out,x=[0,120,240],y=[hue,hue,hue],domain='hue',codomain='hue',\
            r_range=r_range, g_range=g_range,b_range=b_range,c_range=c_range,h_range=h_range,\
            l_range=l_range,u_range=u_range,v_range=v_range,smooth=smooth, mode = 'additive',\
            relative_chroma=False)
    if chroma != 1.0:
        out = _perturb(out,x=[0,100,200],y=[chroma,chroma,chroma],domain='chroma',codomain='chroma',\
            r_range=r_range, g_range=g_range,b_range=b_range,c_range=c_range,h_range=h_range,\
            l_range=l_range,u_range=u_range,v_range=v_range,smooth=smooth, mode = 'multiplicative',\
            relative_chroma=False)
    if bright != 0.0:
        out = _perturb(out,x=[0,100],y=[bright,bright],domain='lightness',codomain='lightness',\
            r_range=r_range, g_range=g_range,b_range=b_range,c_range=c_range,h_range=h_range,\
            l_range=l_range,u_range=u_range,v_range=v_range,smooth=smooth, mode = 'additive',\
            relative_chroma=False)
    return out
