import os
import struct
import sys
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import pprint
from datetime import datetime
import pickle

from azureml.core.datastore import Datastore
from azureml.data.datapath import DataPath
from azureml.core.dataset import Dataset
from azureml.core import Workspace, Environment

from tempfile import TemporaryDirectory

from .SeismicStructure import *


#-----------------------------------------------------------------------------------------------------------------------
l_int = struct.calcsize('i')
l_uint = struct.calcsize('I')
l_long = 4;
l_ulong = struct.calcsize('L')
l_short = struct.calcsize('h')
l_ushort = struct.calcsize('H')
l_char = struct.calcsize('c')
l_uchar = struct.calcsize('B')
l_float = struct.calcsize('f')
#-----------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------------------

def getSegyHeader(filename, endian, SH_def, start_index):  # modified by A Squelch
    """
    SegyHeader=getSegyHeader(filename)
    """

    data = open(filename,'rb').read()

    SegyHeader = {'filename': filename}

    j = 0;
    for key in SH_def.keys():
        j = j + 1;
        pos = SH_def[key]["pos"]
        pos = pos-1 # This was a change done by abhimanyu to make it more human readable
        format = SH_def[key]["type"]

        SegyHeader[key], index = getValue(data, pos, format, endian);


    # SET NUMBER OF BYTES PER DATA SAMPLE
    bps = getBytePerSample(SH= SegyHeader, SH_def= SH_def)

    filesize = len(data)
    no_extended_headers = SegyHeader["NumberOfExtTextualHeaders"]
    ntraces = (filesize - 3600-start_index -3200*no_extended_headers) / (SegyHeader['ns'] * bps + 240) # This needs to be modified more
    SegyHeader["ntraces"] = ntraces
    SegyHeader["time"] = np.arange(SegyHeader['ns']) * SegyHeader['dt'] / 1e+6
    SegyHeader['bps'] = bps
    

    return SegyHeader

def getValue(data, index, ctype='l', endian='>', number=1):
    """
    getValue(data,index,ctype,endian,number)
    """
    if (ctype == 'l') | (ctype == 'long') | (ctype == 'int32'):
        size = l_long
        ctype = 'l'
    elif (ctype == 'L') | (ctype == 'ulong') | (ctype == 'uint32'):
        size = l_ulong
        ctype = 'L'
    elif (ctype == 'h') | (ctype == 'short') | (ctype == 'int16'):
        size = l_short
        ctype = 'h'
    elif (ctype == 'H') | (ctype == 'ushort') | (ctype == 'uint16'):
        size = l_ushort
        ctype = 'H'
    elif (ctype == 'c') | (ctype == 'char'):
        size = l_char
        ctype = 'c'
    elif (ctype == 'B') | (ctype == 'uchar'):
        size = l_uchar
        ctype = 'B'
    elif (ctype == 'f') | (ctype == 'float'):
        size = l_float
        ctype = 'f'
    elif (ctype == 'ibm'):
        size = l_float
        ctype = 'ibm'
    else:
        print('Bad Ctype : ' + ctype, -1)

    index_end = index + size * number

    if (ctype == 'ibm'):
        # ASSUME IBM FLOAT DATA
        Value = list(range(int(number)))
        for i in np.arange(number):
            index_ibm_start = i * 4 + index
            index_ibm_end = index_ibm_start + 4;
            ibm_val = ibm2ieee2(data[index_ibm_start:index_ibm_end])
            Value[i] = ibm_val;
        # this resturn an array as opposed to a tuple
    else:
        # ALL OTHER TYPES OF DATA
        cformat = 'f' * number
        cformat = endian + ctype * number


        Value = struct.unpack(cformat, data[index:index_end])

    if (ctype == 'B'):
        print('getValue : Ineficient use of 1byte Integer...', -1)

        vtxt = 'getValue : ' + 'start=' + str(index) + ' size=' + str(size) + ' number=' + str(
            number) + ' Value=' + str(Value) + ' cformat=' + str(cformat)
        print(vtxt, 20)

    if number == 1:
        return Value[0], index_end
    else:
        return Value, index_end

def getSegyTraceHeadersDataframe(start_index, bin_header, STH_def, endian, filename):

    ntraces = int(bin_header['ntraces'])
    bps = int(bin_header['bps'])
    no_ext_hdr = int(bin_header['NumberOfExtTextualHeaders'])

    columns = []
    index = []
    for key in STH_def.keys():
        columns.append(key)
    for i in range(1,ntraces+ 1):
        index.append(i)
    trchdr_df = pd.DataFrame(index=index, columns=columns)

    data = open(filename,'rb').read()

   
    key_list = [] 
    for key in STH_def.keys():
        key_list.append(key)
        
    for i in tqdm(range(0,len(key_list))):
        key = key_list[i]
        THpos = STH_def[key]["pos"]
        THformat = STH_def[key]["type"]
        #print('Extracting key: ' + key)
        for itrace in range(1, ntraces + 1):
        #for itrace in range(1, 3):
            pos = THpos + 3600 + (bin_header["ns"] * bps + 240) * (itrace - 1) + start_index + 3200*no_ext_hdr;
            pos = pos-1 #This is the change made for making trace pos relateable
            trchdr_df.at[itrace,key], index = getValue(data, pos, THformat, endian, 1)

    return trchdr_df


def ibm2Ieee(ibm_float):
    """
    ibm2Ieee(ibm_float)
    Used by permission
    (C) Secchi Angelo
    with thanks to Howard Lightstone and Anton Vredegoor.
    """
    """    
    I = struct.unpack('>I',ibm_float)[0]
    sign = [1,-1][bool(i & 0x100000000L)]
    characteristic = ((i >> 24) & 0x7f) - 64
    fraction = (i & 0xffffff)/float(0x1000000L)
    return sign*16**characteristic*fraction
    """

def ibm2ieee2(ibm_float):
    """
    ibm2ieee2(ibm_float)
    Used by permission
    (C) Secchi Angelo
    with thanks to Howard Lightstone and Anton Vredegoor.
    """
    dividend = float(16 ** 6)

    if ibm_float == 0:
        return 0.0
    istic, a, b, c = struct.unpack('>BBBB', ibm_float)
    if istic >= 128:
        sign = -1.0
        istic = istic - 128
    else:
        sign = 1.0
    mant = float(a << 16) + float(b << 8) + float(c)
    return sign * 16 ** (istic - 64) * (mant / dividend)

def getBytePerSample(SH,SH_def):
    revision = SH["SegyFormatRevisionNumber"]
    if (revision == 100):
        revision = 1
    if (revision == 256):  # added by A Squelch
        revision = 1

    dsf = SH["DataSampleFormat"]

    try:  # block added by A Squelch
        bps = SH_def["DataSampleFormat"]["bps"][revision][dsf]
    except KeyError:
        print("")
        print("  An error has ocurred interpreting a SEGY binary header key")
        print("  Please check the Endian setting for this file: ", SH["filename"])
        sys.exit()
    return bps

#-----------------------------------------------------------------------------------------------------------------------
#Sub classes
#-----------------------------------------------------------------------------------------------------------------------
class Axis(object):
    def __init__(self):
        self.name = ''
        self.min = 0
        self.max = 0
        self.sort_order = 'asc'
        self.step = 1
        self.unit = None
        self.step_size = None
        self.number = 0

class BinHeader(object):
    def __init__(self, bin_def_dict):
        self.definition = bin_def_dict
        self.data = None
        self.loaded = False

class TrcHeader(object):
    def __init__(self,trc_def_dict):
        self.definition = trc_def_dict
        self.data = None
        self.loaded = False


class SegyData3D(object):
    def __init__(self, sort_list):

        self.traces_loaded = False

        self.sort_list = sort_list

        self.space_dims = 2
        self.total_dims = self.space_dims + 1

        self.axis = []

        for i in range(0,self.total_dims):
            self.axis.append(copy.deepcopy(Axis()))

        self.trace_map = None
        self.hit_map = None
        self.grid = None
        self.geom_loaded = False

    def set_geometry(self,trc,bin):
        print('Calculating Extents for Geometry')
        for i in range (0,self.space_dims):
            for key in self.sort_list[i].keys():
                setattr(self.axis[i], key, self.sort_list[i][key])
            self.axis[i].max = trc[self.axis[i].name].max()
            self.axis[i].min = trc[self.axis[i].name].min()
            self.axis[i].number = self.axis[i].max - self.axis[i].min + 1
        print('Done...')
        

        self.axis[-1].name = 'Time'
        self.axis[-1].step_size = 4
        self.axis[-1].unit = 'ms'
        self.axis[-1].number = bin['ns']



    # space units & step size
    # Time units & unit size

    def create_maps(self,trc,bin):
        print('Creating Maps')
        self.hit_map =np.zeros((self.axis[0].number,self.axis[1].number))
        self.trace_map = np.zeros((self.axis[0].number, self.axis[1].number))
        for i in range(1,int(bin['ntraces'])+1):
            dim0_index = trc.loc[i,self.axis[0].name] - self.axis[0].min
            dim1_index = trc.loc[i,self.axis[1].name] - self.axis[1].min
            self.hit_map[dim0_index,dim1_index] = 1
            self.trace_map[dim0_index, dim1_index] = i

        print('Done...')


    def read_geometry(self,trc,bin):
        if self.geom_loaded is False:
            print('Creating trace geometry')
            self.set_geometry(trc=trc, bin=bin)
            self.create_maps(trc=trc,bin=bin)
            self.geom_loaded = True
            print('Geometry created')
            print('Done...')

    def read_data(self,trc,bin,endian,filename,start_index, no_traces):
        self.read_geometry(trc=trc,bin=bin)
        self.data = np.zeros((self.axis[0].number, self.axis[1].number, self.axis[2].number))
        data = open(filename, 'rb').read()
        self.dsf = None
        if (bin["DataSampleFormat"] == 1):
           self.dsf = 'ibm'
        elif (bin["DataSampleFormat"] == 2):
            self.dsf = 'l'
        elif (bin["DataSampleFormat"] == 3):
            self.dsf ='h'
        elif (bin["DataSampleFormat"] == 5):
            self.dsf ='float'
        elif (bin["DataSampleFormat"] == 8):
            self.dsf = 'B'
        else:
            print("readSegyData : DSF=" + str(bin["DataSampleFormat"]) + ", NOT SUPORTED")
            
        if no_traces == 0:
            no_traces = int(bin['ntraces'])

        print('Reading all traces')

        for i in tqdm(range(1,no_traces +1)):
            index = 3600 + (i - 1) * (240 + bin['ns'] * bin['bps']) + 240 + start_index + 3200*bin['NumberOfExtTextualHeaders']
            dim0_index = trc.loc[i, self.axis[0].name] - self.axis[0].min
            dim1_index = trc.loc[i, self.axis[1].name] - self.axis[1].min
            self.data[dim0_index, dim1_index,:] = getValue(data, index, self.dsf, endian, bin['ns'])[0]

        print('Done...')

#-----------------------------------------------------------------------------------------------------------------------
#SegyObj class
#-----------------------------------------------------------------------------------------------------------------------
class SegyObj(object):
 
    def __init__(self, file_name, file_path, bin_def_dict, trc_def_dict, segy_type, tape_label, endian, sort_list_dict):
       
        self.type = segy_type
        self.file = file_path
        self.filename = file_name

        self.bin = BinHeader(bin_def_dict)
        self.trc = TrcHeader(trc_def_dict)


        if self.type =='3D':
            self.traces = SegyData3D(sort_list_dict)
           


        self.tape_label = tape_label
        self.endian = endian      

        if self.tape_label is True:
            self.start_index = 128
        else:
            self.start_index = 0

    def read_bin_headers(self, show_summary):
        """
        Reads the binary trace header

        :return: None
        """

        if self.bin.loaded is False:
            if os.path.exists(self.file):

                print('Reading Binary header')

                self.bin.data = getSegyHeader(filename=self.file, endian=self.endian, SH_def=self.bin.definition,
                                          start_index=self.start_index)
                
                if show_summary is True:
                
                    for key in self.bin.data.keys():
                        if key not in ['time']:
                            print('Binary header : {:30} => {}'.format(key, str(self.bin.data[key])))
                
                self.bin.loaded = True

                print('Done...')

    def read_trc_headers(self, show_summary):
        """
        Reads the trace headers defined in self.def_key_list is self.load_all_headers is False
        otherwise reads all the trace headers in self.trc.definition

        :return: None
        """


        if self.trc.loaded is False:
            
            print('Reading trace headers...')
            self.trc.data = getSegyTraceHeadersDataframe(filename=self.file, start_index=self.start_index,
                                                         endian=self.endian,
                                                         bin_header=self.bin.data, STH_def=self.trc.definition)
            
            if show_summary is True:
            
                for key in self.trc.definition.keys():
                    print('Trace Header : {:45} :min = {:10} :max = {}'.format(key, str(self.trc.data[key].min()), str(self.trc.data[key].max()) ) )              


            self.trc.loaded = True


            print('Done...')


    def read_traces(self, show_summary, no_traces):

        self.traces.read_geometry(bin=self.bin.data, trc=self.trc.data)

        self.traces.read_data(bin=self.bin.data,trc=self.trc.data,endian=self.endian,
                                    filename=self.file,start_index=self.start_index, no_traces = no_traces)

    def read(self, read_traces = False, show_summary = True, no_traces = 0):
        """
        Reads all the trace headers and all the data

        :return: None
        """
        self.read_bin_headers(show_summary)
        self.read_trc_headers(show_summary)
        
        if read_traces ==True:
            self.read_traces(show_summary, no_traces)



def ReadSegy(job_params_dict):
    with TemporaryDirectory() as path:
        
        blobstore_name = job_params_dict['blobstore_name']
        folder_path = job_params_dict['folder_path']
        segy_file_name = job_params_dict['segy_file_name']
        segy_local_path = os.path.join(path, segy_file_name)
        bin_def_dict = job_params_dict['bin_def_dict']
        trc_def_dict = job_params_dict['trc_def_dict']
        segy_type = job_params_dict['segy_type']
        tape_label = job_params_dict['tape_label']
        endian = job_params_dict['endian']
        sort_list_dict = job_params_dict['sort_list_dict']
        read_traces = job_params_dict['read_traces']
        show_summary = job_params_dict['show_summary']
        no_traces = job_params_dict['no_traces']

        
        workspace = Workspace.from_config()
        datastore = Datastore.get(workspace, blobstore_name)
        data_path = DataPath(datastore, folder_path)
        dataset_full = Dataset.File.from_files(path=data_path)
        mount_context = dataset_full.mount(path)
        mount_context.start()

        
        segy_obj = SegyObj(file_name = segy_file_name, file_path = segy_local_path, bin_def_dict=bin_def_dict, trc_def_dict = trc_def_dict,
                      segy_type = segy_type,  tape_label = tape_label, 
                       endian=endian, sort_list_dict = sort_list_dict)

        segy_obj.read(read_traces = read_traces, show_summary = show_summary, no_traces=no_traces)
        
        mount_context.stop()
        
        return segy_obj


def local_file_path(root, relative):
    path = os.path.join(root, relative)
    if os.path.exists(path):
        os.remove(path)
        print('Deleting Existing path: ' + path)
    return path

def create_attributes_dict(segy_obj):
    now = datetime.now()
    
    attributes_dict = {
        'segyfile' : segy_obj.filename,
        'datatype' : segy_obj.type,
        'created'  : now.strftime("%m/%d/%Y, %H:%M:%S")
    }
    
    return attributes_dict

def create_geometry_dict(segy_obj):
    geometry_dict = {'sort_list' : segy_obj.traces.sort_list}
    for i in range(0, segy_obj.traces.total_dims):
        geometry_dict[segy_obj.traces.axis[i].name] = {
            'min'       : segy_obj.traces.axis[i].min,
            'max'       : segy_obj.traces.axis[i].max,
            'sort_order': segy_obj.traces.axis[i].sort_order,
            'step'      : segy_obj.traces.axis[i].step,
            'unit'      : segy_obj.traces.axis[i].unit,
            'step_size' : segy_obj.traces.axis[i].step_size,
            'number'    : segy_obj.traces.axis[i].number,
        } 
        
    return geometry_dict

    

def CreateLocalSegyObject(job_params_dict, segy_obj = None):
    
    local_path_list = []
    
    # Now check if the output directory exists
    local_root = job_params_dict['local_root']
    job_name= job_params_dict['job_name']
    output_file_name = job_params_dict['output_file_name']
    
    overwrite = job_params_dict['overwrite']
    
    local_job_root = os.path.join(local_root, job_name, output_file_name)
    
    if os.path.exists(local_job_root):
        print('Local files already exist')
        if overwrite is False:
            sys.exit('Aborting.. please use overwirte = True')
        else:
            print('All cloud and local files will be overwritten!!!')
    
    os.makedirs(local_job_root, exist_ok = True)
    #Read the segy file and crete the in memory object
    
    if segy_obj is None:
        segy_obj = ReadSegy(job_params_dict)
    
    #save attribute of file
    print('Saving: File attributes as pickle')
    attributes_path = local_file_path(local_job_root, attributes_name)
    attributes_dict = create_attributes_dict(segy_obj)
    with open(attributes_path, 'wb') as file:
        pickle.dump(attributes_dict, file)
    local_path_list.append(attributes_path)
    print('Done...')
    
    #save binary header definiton to file
    print('Saving: Binary header definition as pickle')
    bin_header_def_path = local_file_path(local_job_root, bin_hdr_def_name)
    with open(bin_header_def_path, 'wb') as file:
        pickle.dump(segy_obj.bin.definition, file)
    local_path_list.append(bin_header_def_path)
    print('Done...')
    
    #save binary header data to file
    print('Saving: Binary header Data as pickle')
    bin_data_path = local_file_path(local_job_root, bin_data_name)
    with open(bin_data_path, 'wb') as file:
        pickle.dump(segy_obj.bin.data, file)
    local_path_list.append(bin_data_path)
    print('Done...')
    
    #save trace header definiton of file
    print('Saving: Trace header definition as pickle')
    trc_header_def_path = local_file_path(local_job_root, trc_hdr_def_name)
    with open(trc_header_def_path, 'wb') as file:
        pickle.dump(segy_obj.trc.definition, file)
    local_path_list.append(trc_header_def_path)
    print('Done...')
    
    #save binary header data to file
    print('Saving: Trace header Data as csv')
    trc_data_path = local_file_path(local_job_root, trc_data_name)
    segy_obj.trc.data.to_csv(trc_data_path)
    local_path_list.append(trc_data_path)
    print('Done...')
    
    #save geometry of file
    print('Saving: Geometry definition as pickle')
    geometry_dict = create_geometry_dict(segy_obj)
    geometry_path = local_file_path(local_job_root, geometry_name)
    with open(geometry_path, 'wb') as file:
        pickle.dump(geometry_dict, file)
    local_path_list.append(geometry_path)
    print('Done...')
    
    #save hitmap
    print('Saving: hitmap sa numpy array')
    hitmap_path = local_file_path(local_job_root, hitmap_name)
    np.save(file=hitmap_path, arr = segy_obj.traces.hit_map)
    local_path_list.append(hitmap_path)
    print('Done ...')
    
    #save tarcemap
    print('Saving: tracemap sa numpy array')
    tracemap_path = local_file_path(local_job_root, tracemap_name)
    np.save(file=tracemap_path, arr = segy_obj.traces.trace_map)
    local_path_list.append(tracemap_path)
    print('Done ...')
    
    #Save trace data
    print('Saving: Trace data as numpy array')
    trace_data_path = local_file_path(local_job_root, trace_name)
    np.save(file=trace_data_path, arr = segy_obj.traces.data)
    local_path_list.append(trace_data_path)
    print('Done...')
    
    return local_path_list


def LoadSegy2Blob(job_params_dict, local_path_list = None):
    
    workspace = Workspace.from_config()
    blobstore_name = job_params_dict['blobstore_name']
    
    datastore = Datastore(workspace, blobstore_name)
    
    if local_path_list is None:
        local_path_list = CreateLocalSegyObject(job_params_dict= job_params_dict, segy_obj = None)
        
    output_file_name = job_params_dict['output_file_name']
        
    target_path = output_file_name
    
    datastore.upload_files(files = local_path_list , target_path= output_file_name, show_progress = True, overwrite = True)
    
    if job_params_dict['remove_local'] is True:
        for path in local_path_list:
            print(f'Deleting local file : {path}')
            os.remove(path)

    

    

    

