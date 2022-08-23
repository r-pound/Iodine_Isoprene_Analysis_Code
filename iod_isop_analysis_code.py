import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 16})
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
warnings.filterwarnings('ignore')
import cartopy.feature as cfeature
#############################################################################
#                               Open data files
#############################################################################
def model_data():

    """
    open species conc, conc after chem and meterology for analysis, cutting out the
    spin up period.
    """

    root_wd = 'GC/rundirs/13/'
    base_wd = 'merra2_2x25_standard/OutputDir/'
    iod_dep_wd = 'merra2_2x25_standard_noIdep/OutputDir/'
    iod_wd = 'merra2_2x25_standard_noI/OutputDir/'
    isop_wd = 'merra2_2x25_standard_noISOP/OutputDir/'

    specs = 'GEOSChem.SpeciesConc.201*_0000z.nc4'#2016
    cac = 'GEOSChem.ConcAfterChem.201*_0000z.nc4'
    met = 'GEOSChem.StateMet.201*_0000z.nc4'

    #open up the files then return
    bse_spec = xr.open_mfdataset(root_wd+base_wd+specs,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
    bse_cac  = xr.open_mfdataset(root_wd+base_wd+cac,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
    bse_met  = xr.open_mfdataset(root_wd+base_wd+met,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))

    iod_spec = xr.open_mfdataset(root_wd+iod_wd+specs,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
    iod_cac  = xr.open_mfdataset(root_wd+iod_wd+cac,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
    iod_met  = xr.open_mfdataset(root_wd+iod_wd+met,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))

    isp_spec = xr.open_mfdataset(root_wd+isop_wd+specs,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
    isp_cac  = xr.open_mfdataset(root_wd+isop_wd+cac,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
    isp_met  = xr.open_mfdataset(root_wd+isop_wd+met,combine='by_coords')\
                 .sel(time=slice('2016-07-01','2017-07-01'))
return
#############################################################################
#                                Metrics
#############################################################################
def burden(concs,met):

    """
    calculate the global tropospheric and 0-1km burden of ozone from the Met and
     Species conc diagnostics, input in daily averages. Output in Tg/yr
    """

    names = ['O3']
    mol_weight = [48]
    levs = met['Met_TropLev'].sel(time=slice('2016-07-01','2017-07-01')).values #2016-2017
    mask = np.zeros([len(met.time.values),47,91,144])
    for t in range(len(met.time.values)):
        for y in range(91):
            for x in range(144):
                mask[t,int(levs[t,y,x]):47,y,x] = 1
    print('mask made')

    airden = met['Met_AD'].sel(time=slice('2016-07-01','2017-07-01')).where(mask==0).values
    val = concs[f'SpeciesConc_{names[0]}'].sel(time=slice('2016-07-01','2017-07-01')).where(mask==0).values
    times = len(concs.time.values)
    val = np.nansum(val*(airden*1E3/28.88)*mol_weight[0])*1E-12/times
    print(f'{names[0]} = {val}')

    # O3 in 0-1km
    airden = met['Met_AD'].sel(time=slice('2016-07-01','2017-07-01')).values[:,0:8,:,:]
    val = concs[f'SpeciesConc_O3'].sel(time=slice('2016-07-01','2017-07-01')).values[:,0:8,:,:]
    times = len(concs.time.values)
    val = np.nansum(val*(airden*1E3/28.88)*mol_weight[0])*1E-12/times
    print(f'{names[0]} = {val}')
return
#############################################################################
def massweight_OH(cac,met):

    """
    calculate global mass weighted tropospheric OH from daily average Met
     and conc after chem diagnostics
    """

    levs = met['Met_TropLev'].sel(time=slice('2016-07-01','2017-07-01')).values
    mask = np.zeros([len(met.time.values),47,91,144])
    for t in range(len(met.time.values)):
        for y in range(91):
            for x in range(144):
                mask[t,int(levs[t,y,x]):47,y,x] = 1
    print('mask made')

    OH = cac['OHconcAfterChem'].sel(time=slice('2016-07-01','2017-07-01')).\
                                   where(mask==0).values

    airden = met['Met_AD'].sel(time=slice('2016-07-01','2017-07-01')).\
                          where(mask==0).values

    weighted = np.nansum(OH * airden)/np.nansum(airden)*1e-5
    print(weighted)

    temp = met['Met_T'].where(mask==0).mean(dim='time').values
    CH4 = (2.45E-12 * np.ma.exp((-1775./temp)))*OH
    CH4 = 1./(np.nansum(CH4*airden)/np.nansum(airden)) /60./60./24./365.

    print(f'methane lifetime in years {CH4}')
    return
############################################################################
def iod_isop_emission_delta():

    """
    Calculate the seasonal variation in iodine and isoprene emissions
    """
    ds = xr.open_mfdataset(\
          'GC/rundirs/13/merra2_2x25_standard/OutputDir/HEMCO_diagnostics.201*',\
          combine='by_coords').sel(time=slice('2016-07-01','2017-07-01'))

    iodine = ds['InvIODINE_I2'].sel(lat=slice(0,90)).sum(dim='lat').sum(dim='lon').values + \
             ds['InvIODINE_HOI'].sel(lat=slice(0,90)).sum(dim='lat').sum(dim='lon').values
    isoprene = ds['EmisISOP_Total'].sel(lat=slice(0,90)).sum(dim='lev').sum(dim='lat').sum(dim='lon').values

    iod_delta = (np.amax(iodine)-np.amin(iodine))/np.amin(iodine)*100
    iso_delta = (np.amax(isoprene)-np.amin(isoprene))/np.amin(isoprene)*100
    return
#############################################################################
#                                   plots
#############################################################################
def per_change_year(base,iod,isop,spec,units,multi=1e9,\
                        wd='plots/iod-isop/'):
    matplotlib.rcParams.update({'font.size': 22})
    """
    create percentage and absolute year average changes in surface and
     verticle for supplied species OH and O3 profiles ( and CO, NOy, NOx)
    """
    NOy = ['NO','NO2','NO3','HNO2','HNO3','HNO4','BrNO2','BrNO3','ClNO2','ClNO3','ETHLN',\
           'ETNO3','HONIT','ICN','IDN','IHN1','IHN2','IHN3','IHN4','INPB','INPD','ITCN',\
           'ITHN','MCRHN','MCRHNB','MENO3','MONITA','MONITS','MONITU','MPAN','MPN','MVKN',\
           'N2O5','NPRNO3','PAN','PPN','PROPNN','R4N2','IONO','IONO2']

    x = base.lon.values
    y = base.lat.values
    v = [0.058,0.189,0.32,0.454,0.589,0.72,0.864,1.004,1.146,1.29,1.436,1.584,\
         1.759,1.988,2.249,2.517,2.792,3.074,3.439,3.896,4.375,4.879,5.413,\
         5.98,6.585,7.237,7.943,8.846,9.936]
    t_v = [0.058,0.189,0.32,0.454,0.589,0.72,0.864,1.004,1.146,1.29,1.436,1.584,\
         1.759,1.988,2.249,2.517,2.792,3.074,3.439,3.896,4.375,4.879,5.413,\
         5.98,6.585,7.237,7.943,8.846,9.936,11.021,12.086,13.134,14.170,15.198,\
         16.222,17.243,18.727,20.836,23.020,25.307,28.654,34.024,40.166,47.135,\
         54.834,63.053,72.180]

    x = [x,x,y,y]
    y = [y,y,v,v]

    surf_ticks = np.array([-30, -25, -20, -15, -10,  -5,  -2,    2,    5,  10,  15,  20,  25,  30])
    #surf_ticks = np.arange(-80,20,10)
    vert_ticks = np.array([-15, -12,  -9,  -6,  -3,   -1,   1,   3,   6,   9,  12,  15])
    #vert_ticks = np.arange(-20,20,5)
    surf_ticks = np.array([-85,-75,-65,-55,-45,-35,-25,-15,-5,5,15])#OH
    surf_ticks_l = [-85,-65,-45,-25,-15,-5,5,15]
    vert_ticks = np.array([-25, -20, -15, -10,  -5,  -2,   2, 5])#OH
    surf_ticks = [-5,-1,1,5,10,15,20,25,30,35,40] #CO
    vert_ticks = [-5,-1,1,5,10,15,20,25,30,35,40] #CO
    #nr_map = np.vstack([plt.cm.Blues_r(np.linspace(0,1,7))[0:6],np.array([[1,1,1,1],[0.99358708,0.83234141,0.76249135,1]])])
    surf_ticks = [-15,-10,-5,-1,1,5,10,15,20,25]# NOY surf - -15 25
    vert_ticks = [-5,-1,1,5,10,15,20,25]# NOY vert - -5  25
    surf_ticks = np.arange(-30,30) #NOx
    vert_ticks = np.arange(-15,15) #NOx

    na_map = np.array([[0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [1.        , 1.        , 1.        , 1.        ],\
                       [0.98823529, 0.62614379, 0.50849673, 1.        ],\
                       [0.98357555, 0.41279508, 0.28835063, 1.        ],\
                       [0.89019608, 0.18562092, 0.15294118, 1.        ],\
                       [0.69439446, 0.0700346 , 0.09231834, 1.        ],\
                       [0.52      , 0.02      , 0.07,       1.        ],\
                       [0.40392157, 0.        , 0.05098039, 1.        ]])
    na_map = np.array([[0.21568627, 0.52941176, 0.75424837, 1.        ],\
                       [0.21568627, 0.52941176, 0.75424837, 1.        ],\
                       [0.4227451 , 0.68407536, 0.83989235, 1.        ],\
                       [0.4227451 , 0.68407536, 0.83989235, 1.        ],\
                       [0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [1.    , 1.    , 1.    , 1.        ],\
                       [0.98823529, 0.62614379, 0.50849673, 1.        ],\
                       [0.98823529, 0.62614379, 0.50849673, 1.        ],\
                       [0.98357555, 0.41279508, 0.28835063, 1.        ],\
                       [0.98357555, 0.41279508, 0.28835063, 1.        ],\
                       [0.89019608, 0.18562092, 0.15294118, 1.        ],\
                       [0.89019608, 0.18562092, 0.15294118, 1.        ],\
                       [0.69439446, 0.0700346 , 0.09231834, 1.        ],\
                       [0.69439446, 0.0700346 , 0.09231834, 1.        ],\
                       [0.40392157, 0.        , 0.05098039, 1.        ],\
                       [0.40392157, 0.        , 0.05098039, 1.        ]])
    # OH colourmap
    nr_map = np.array([[0.03137255, 0.18823529, 0.41960784, 1.        ],\
                       [0.03137255, 0.18823529, 0.41960784, 1.        ],\
                       [0.06251442, 0.35750865, 0.64290657, 1.        ],\
                       [0.06251442, 0.35750865, 0.64290657, 1.        ],\
                       [0.21568627, 0.52941176, 0.75424837, 1.        ],\
                       [0.21568627, 0.52941176, 0.75424837, 1.        ],\
                       [0.4227451 , 0.68407536, 0.83989235, 1.        ],\
                       [0.4227451 , 0.68407536, 0.83989235, 1.        ],\
                       [0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [1.    , 1.    , 1.    , 1.        ],\
                       [0.99358708, 0.83234141, 0.76249135, 1.        ]])
    nr_map = np.array([[0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [1.        , 1.        , 1.        , 1.        ],\
                       [0.98823529, 0.62614379, 0.50849673, 1.        ],\
                       [0.98823529, 0.62614379, 0.50849673, 1.        ],\
                       [0.98357555, 0.41279508, 0.28835063, 1.        ],\
                       [0.98357555, 0.41279508, 0.28835063, 1.        ],\
                       [0.89019608, 0.18562092, 0.15294118, 1.        ],\
                       [0.89019608, 0.18562092, 0.15294118, 1.        ],\
                       [0.69439446, 0.0700346 , 0.09231834, 1.        ],\
                       [0.69439446, 0.0700346 , 0.09231834, 1.        ],\
                       [0.40392157, 0.        , 0.05098039, 1.        ],\
                       [0.40392157, 0.        , 0.05098039, 1.        ]])
    nr_map = matplotlib.colors.ListedColormap(nr_map)
    nr_map.set_under(np.array([0.01137255,0.18823529,0.51960784,1.]))
    nr_map.set_over(np.array([0.40392157,0.,0.05098039,1.]))

    nb_map = np.array([[0.03137255, 0.18823529, 0.41960784, 1.        ],\
                       [0.03137255, 0.18823529, 0.41960784, 1.        ],\
                       [0.06251442, 0.35750865, 0.64290657, 1.        ],\
                       [0.21568627, 0.52941176, 0.75424837, 1.        ],\
                       [0.4227451 , 0.68407536, 0.83989235, 1.        ],\
                       [0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [1.    , 1.    , 1.    , 1.        ],\
                       [0.99358708, 0.83234141, 0.76249135, 1.        ]])
    nb_map = np.array([[0.67189542, 0.81437908, 0.90065359, 1.        ],\
                       [1.        , 1.        , 1.        , 1.        ],\
                       [0.98823529, 0.62614379, 0.50849673, 1.        ],\
                       [0.98357555, 0.41279508, 0.28835063, 1.        ],\
                       [0.89019608, 0.18562092, 0.15294118, 1.        ],\
                       [0.69439446, 0.0700346 , 0.09231834, 1.        ],\
                       [0.40392157, 0.        , 0.05098039, 1.        ]])
    nb_map = matplotlib.colors.ListedColormap(nb_map)
    nb_map.set_under(np.array([0.01137255,0.18823529,0.51960784,1.]))
    nb_map.set_over(np.array([0.40392157,0.,0.05098039,1.]))

    tracer = f'SpeciesConc_O3'
    #tracer = f'OHconcAfterChem' 128
    cmap1 = np.vstack((matplotlib.cm.get_cmap('Blues_r',8)(range(8))[1:],\
                     np.array([1,1,1,1]),\
                     matplotlib.cm.get_cmap('Reds',8)(range(8))[1:]))
    cmap1 = colors.ListedColormap(cmap1)
    cmap2 = np.vstack((matplotlib.cm.get_cmap('Blues_r',6)(range(6))[1:],\
                     np.array([1,1,1,1]),\
                     matplotlib.cm.get_cmap('Reds',6)(range(6))[1:]))
    cmap2 = colors.ListedColormap(cmap2)


    b = np.zeros((91,144))
    n = np.zeros((91,144))
    n2 = np.zeros((91,144))
    for i in NOy:
         b = b + base[f'SpeciesConc_{i}'].sel(lev=base.lev.values[0]).mean(dim='time').values
         n = n + iod[f'SpeciesConc_{i}'].sel(lev=iod.lev.values[0]).mean(dim='time').values
         n2 = n2 + isop[f'SpeciesConc_{i}'].sel(lev=isop.lev.values[0]).mean(dim='time').values
    iod_surf = (-n + b)/b * 100
    isop_surf = (-n2 + b)/b * 100

    b = base[tracer].sel(lev=base.lev.values[0]).mean(dim='time').values\
           *multi
    n  = iod[tracer].sel(lev=iod.lev.values[0]).mean(dim='time').values\
           *multi
    iod_surf = (-n + b)/b * 100
    n  = isop[tracer].sel(lev=isop.lev.values[0]).mean(dim='time').values\
           *multi
    isop_surf = (-n + b)/b * 100

    b = np.zeros((29,91))
    n = np.zeros((29,91))
    n2 = np.zeros((29,91))
    for i in NOy:
         b = b + base[f'SpeciesConc_{i}'].mean(dim='time').mean(dim='lon').values[0:29,:]
         n = n + iod[f'SpeciesConc_{i}'].mean(dim='time').mean(dim='lon').values[0:29,:]
         n2 = n2 + isop[f'SpeciesConc_{i}'].mean(dim='time').mean(dim='lon').values[0:29,:]
    iod_vert = (-n + b)/b * 100
    isop_vert = (-n2 + b)/b * 100

    b = base[tracer].mean(dim='time').mean(dim='lon').values[0:29,:]*multi
    n  = iod[tracer].mean(dim='time').mean(dim='lon').values[0:29,:]*multi
    iod_vert = (-n + b)/b * 100
    n  = isop[tracer].mean(dim='time').mean(dim='lon').values[0:29,:]*multi
    isop_vert = (-n + b)/b * 100

    fig = plt.figure(figsize=[24,16])
    axes=[plt.subplot(2,2,1,projection=ccrs.EqualEarth()),\
          plt.subplot(2,2,2,projection=ccrs.EqualEarth()),\
          plt.subplot(2,2,3),plt.subplot(2,2,4)]

    z = [-1*iod_surf,isop_surf,-1*iod_vert,isop_vert]
    ticks = [surf_ticks,surf_ticks,vert_ticks,vert_ticks]

    for i in range(4):
        norm = colors.Normalize(np.amin(ticks[i]),np.amax(ticks[i]))
        if i == 0 or i == 1:
            axes[i].coastlines()
            csur = axes[i].contourf(x[i],y[i],z[i],ticks[i],extend='both',\
                          transform=ccrs.PlateCarree(),cmap=cmap1,norm=norm)
        else:
            ccol = axes[i].contourf(x[i],y[i],z[i],ticks[i],\
                             cmap=cmap2,extend='both',norm = norm)
            axes[i].set_xlabel('Latitude')
            axes[i].set_ylabel('Altitude [km]')
            axes[i].set_ylim(0,10)

    axes[0].set_title(f'Decrease in O3 from iodine')
    axes[1].set_title(f'Increase in O3 from isoprene')

    cticks = [-30,-20,-10,-5,0,5,10,20,30]
    cax = plt.gcf().add_axes([0.94, 0.5, 0.01, 0.4])
    cbar = plt.colorbar(csur,cax=cax,label='Surface Change [%]',ticks=cticks,\
                        extend='both')
    cticks = np.arange(-15,20,5)
    cax = plt.gcf().add_axes([0.94, 0.05, 0.01, 0.4])
    cbar = plt.colorbar(ccol,cax=cax,label='Zonal Change [%]',ticks=cticks,\
                        extend='both')

    plt.gcf().savefig(wd+f"O3_percent_change.png",dpi=300,transparent=False)
    return
#############################################################################
def O3_surface_delta_seasonal(base,iod,isop,spec,units='ppbv',multi=1e9,\
                        wd='/plots/iod-isop/'):
    """
    create the seasonal plots of iodine isoprene ratio for supplimental material
    """

    #turn into quarterly averages
    base = base.sel(lev=base.lev.values[0]).resample(time="QS-DEC").mean()*1E9
    iod = iod.sel(lev=iod.lev.values[0]).resample(time="QS-DEC").mean()*1E9
    isop = isop.sel(lev=isop.lev.values[0]).resample(time="QS-DEC").mean()*1E9

    x = np.append(base.lon.values,180)
    x2 = base.lon.values
    y = base.lat.values

    u_act = bse_met['Met_U10M'].resample(time="QS-DEC").mean().values
    v_act = bse_met['Met_V10M'].resample(time="QS-DEC").mean().values
    unorm = u_act/np.sqrt(u_act**2 + v_act**2)
    vnorm = v_act/np.sqrt(u_act**2 + v_act**2)

    ratio_delta = np.abs(base['SpeciesConc_O3'].values-isop['SpeciesConc_O3'].values)/\
                  np.abs(base['SpeciesConc_O3'].values-iod['SpeciesConc_O3'].values)

    sub_titles = ['DJF','MAM','JJA','SON']
    order = [2,3,0,1]

    plt.figure(figsize=[12,9])#9,15
    axes=[plt.subplot(2,2,1,projection=ccrs.EqualEarth()),\
          plt.subplot(2,2,2,projection=ccrs.EqualEarth()),\
          plt.subplot(2,2,3,projection=ccrs.EqualEarth()),\
          plt.subplot(2,2,4,projection=ccrs.EqualEarth())] #EuroPP
    cmap = 'PRGn'
    norm = colors.LogNorm(vmin=0.1,vmax=10)

    NA = [-140,-50,25,58]
    AS = [66,148,0,50]
    for i in range(4):
        axes[i].coastlines(resolution='50m',zorder=3)
        axes[i].set_global()
        #axes[i].set_extent(NA)
        axes[i].add_feature(cfeature.BORDERS,zorder=4)
        im = axes[i].pcolormesh(x,y,ratio_delta[order[i],:,:],norm=norm,\
                              transform=ccrs.PlateCarree(),cmap=cmap,zorder=2)
        axes[i].contour(x2,y,ratio_delta[order[i],:,:],levels=[1],\
                        transform=ccrs.PlateCarree(),cmap='autumn',zorder=3)
        axes[i].title.set_text(sub_titles[i])

        #SO X and Y need to be 145,91 but only contain ONLY the  X or Y values because of course
        axes[i].quiver(x2,y, unorm[order[i],:,:], vnorm[order[i],:,:], transform=ccrs.PlateCarree(),zorder=5,scale=0.5,scale_units='x')

    cax = plt.gcf().add_axes([0.05, 0.07, 0.9, 0.03])
    cbar = plt.colorbar(im,label="delta O3 from isoprene / delta O3 from iodine", \
        orientation="horizontal",cax=cax,extend='both')
    plt.gcf().add_artist(matplotlib.lines.Line2D([0.5,0.5],[0.07,0.1],color='red'))

    plt.subplots_adjust(wspace=0.1,hspace=0.005)
    plt.tight_layout(pad=1.2)

    plt.gcf().savefig(wd+f"O3_delta_ratio.png",dpi=300)
    return
#############################################################################
def O3_surface_delta_average(base,iod,isop,spec,units='ppbv',multi=1e9,\
                        wd='/plots/iod-isop/'):
    """
    produce the compilation iodine isoprene ratio plots for section 3.3 in paper
    """
    base = base['SpeciesConc_O3'].sel(lev=base.lev.values[0]).mean(dim='time').values*1E9
    iod = iod['SpeciesConc_O3'].sel(lev=iod.lev.values[0]).mean(dim='time').values*1E9
    isop = isop['SpeciesConc_O3'].sel(lev=isop.lev.values[0]).mean(dim='time').values*1E9

    x = np.append(bse_spec.lon.values,180)
    x2 = bse_spec.lon.values
    y = bse_spec.lat.values

    ratio_delta = np.abs(base-isop)/np.abs(base-iod)

    sub_titles = ['Global','North America','Europe','Asia']

    plt.figure(figsize=[15,10])
    axes=[plt.subplot(2,2,1,projection=ccrs.EqualEarth()),\
          plt.subplot(2,2,2,projection=ccrs.PlateCarree()),\
          plt.subplot(2,2,3,projection=ccrs.EuroPP()),\
          plt.subplot(2,2,4,projection=ccrs.PlateCarree())]
    cmap = 'PRGn'
    norm = colors.LogNorm(vmin=0.1,vmax=10)

    NA = [-140,-50,25,58]
    AS = [66,148,0,50]
    extents = [False,True,False,True]
    vals = [[],NA,[],AS]
    for i in range(4):
        axes[i].coastlines(resolution='50m',zorder=3)
        if i > 0:
            axes[i].add_feature(cfeature.BORDERS,zorder=4)
            axes[i].contour(x2,y,ratio_delta,levels=[1],\
                        transform=ccrs.PlateCarree(),cmap='autumn',zorder=3)
        axes[i].title.set_text(sub_titles[i])
        if extents[i]:
            axes[i].set_extent(vals[i])
        im = axes[i].pcolormesh(x,y,ratio_delta,norm=norm,\
                              transform=ccrs.PlateCarree(),cmap=cmap,zorder=2)
    cax = plt.gcf().add_axes([0.05, 0.07, 0.9, 0.03])
    cbar = plt.colorbar(im,label="delta O3 from isoprene / delta O3 from iodine", \
        orientation="horizontal",cax=cax,extend='both')
    plt.gcf().add_artist(matplotlib.lines.Line2D([0.5,0.5],[0.07,0.1],color='red'))

    plt.subplots_adjust(wspace=0.1,hspace=0.005)
    plt.tight_layout(pad=1.2)

    plt.gcf().savefig(wd+f"O3_delta_ratio_average.png",dpi=300)
    return
##########################################################################
