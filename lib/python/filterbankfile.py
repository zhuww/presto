"""
code to handle filterbank files.

Weiwei Zhu
zhuwwpku@gmail.com
2018/03/25
#parts fo the code were adopted from sigpyproc by Ewen Barr
"""
from struct import unpack
import os, math
import numpy as np

header_keys = { 
    "HEADER_START":None,
    "HEADER_END":None,
    "FREQUENCY_START":None,
    "FREQUENCY_END": None,
    "filename": 'str',
    "telescope_id": 'I',
    "telescope": 'str',
    "machine_id": 'I',
    "data_type": 'I',
    "rawdatafile": 'str',
    "source_name": 'str',
    "barycentric": 'I',
    "pulsarcentric": 'I',
    "az_start": 'd',
    "za_start": 'd',
    "src_raj": 'd',
    "src_dej": 'd',
    "tstart": 'd',
    "tsamp": 'd',
    "nbits": 'I',
    "nsamples": 'I',
    "fch1": 'd',
    "foff": 'd',
    "fchannel": 'd',
    "nchans": 'I',
    "nifs": 'I',
    "refdm": 'd',
    "flux": 'd',
    "period": 'd',
    "nbeams": 'I',
    "ibeam": 'I',
    "hdrlen": 'I',
    "pb":"d",
    "ecc":"d",
    "asini":"d",
    "orig_hdrlen": 'I',
    "new_hdrlen": 'I',
    "sampsize": 'I',
    "bandwidth": 'd',
    "fbottom": 'd',
    "ftop": 'd',
    "obs_date": 'str',
    "obs_time": 'str',
    "signed": 'b',
    "accel": 'd',
    "fchannel": 'd'
    }

nbits_to_dtype = {1:"<u1",
                  2:"<u1",
                  4:"<u1",
                  8:"<u1",
                  16:"<u2",
                  32:"<f4"}


def _read_char(f):
    return unpack("b",f.read(1))[0]

def _read_string(f):
    strlen = unpack("I",f.read(4))[0]
    return f.read(strlen)

def _read_int(f):
    return unpack("I",f.read(4))[0]

def _read_double(f):
    return unpack("d",f.read(8))[0]

def radec_to_str(val):
    """Convert Sigproc format RADEC float to a string.

    :param val: Sigproc style RADEC float (eg. 124532.123)
    :type val: float
    
    :returns: 'xx:yy:zz.zzz' format string
    :rtype: :func:`str`
    """
    if val < 0:
        sign = -1
    else:
        sign = 1
    fractional,integral = math.modf(abs(val))
    xx = (integral-(integral%10000))/10000
    yy = ((integral-(integral%100))/100)-xx*100
    zz = integral - 100*yy - 10000*xx + fractional
    zz = "%07.4f"%(zz)
    return "%02d:%02d:%s"%(sign*xx,yy,zz)


class filterbank(object):

    def __init__(self, filename):
        self._file = open(filename, 'r')
        self.header = self.__parseSHeader()
        self.header["filename"] = filename
        self.header["basename"] = os.path.splitext(filename)[0]

    def __parseSHeader(self):
        """Parse the metadata from a Sigproc-style file header.

        :param filename: file containing the header
        :type filename: :func:`str`
        
        :return: observational metadata
        :rtype: :class:`~sigpyproc.Header.Header`
        """
        #f = open(filename,"r")
        f = self._file
        header = {}
        keylen = unpack("I",f.read(4))[0]
        key = f.read(keylen)
        if key != "HEADER_START":
            raise IOError,"File Header is not in sigproc format"
        while True:
            keylen = unpack("I",f.read(4))[0]
            key = f.read(keylen)
            if not key in header_keys:
                print "'%s' not recognised header key"%(key)
                return None

            if key == "fchannel":
                header["fchannel"].append(_read_double(f))
            elif key == "FREQUENCY_END":
                header["fch1"] = max(header["fchannel"]) #largest first
                #print  header["fchannel"]
                #sys.exit(0)
            else:
                if header_keys[key] == "str":
                    header[key] = _read_string(f)
                elif header_keys[key] == "I":
                    header[key] = _read_int(f)
                elif header_keys[key] == "b":
                    header[key] = _read_char(f)
                elif header_keys[key] == "d":
                    header[key] = _read_double(f)
                if key == "HEADER_END":
                    break
                if key == "FREQUENCY_START":
                    header["fchannel"] = []

        header["hdrlen"] = f.tell()
        f.seek(0,2)
        header["filelen"]  = f.tell()
        header["nbytes"] =  header["filelen"]-header["hdrlen"]
        header["nsamples"] = 8*header["nbytes"]/header["nbits"]/header["nchans"]
        header['ra'] = radec_to_str(header['src_raj'])
        header['dec'] = radec_to_str(header['src_dej'])
        if not "fchannel" in header:
            f1 = header['fch1']
            f2 = f1 + (header['nchans'] - 1) * header['foff']
            header['ftop'] = max((f1, f2)) + abs(0.5 * header['foff'])
            header['fbottom'] = min((f1, f2)) - abs(0.5 * header['foff'])
        else:
            f1 = max(header["fchannel"])
            f2 = min(header["fchannel"])
            header['foff'] = (f2  - f1)/(header["nchans"] - 1)
            header['nchans'] = len(header["fchannel"])
            header['ftop'] = max((f1, f2)) + abs(0.5 * header['foff'])
            header['fbottom'] = min((f1, f2)) - abs(0.5 * header['foff'])
        header['dtype'] = nbits_to_dtype[header['nbits']]
        header['sampsize'] = header['nbits'] / 8. * header['nchans']
        #f.seek(0)
        #f.close()
        return header

    def readblock(self, n):
        if self.header['dtype'] == '<u1':
            if self.header['nbits'] == 1:
                nbyte = int(np.ceil(n/8))
                rawdata = np.fromfile(self._file, dtype='u1', count=nbyte)
                size = rawdata.size
                rawdata = rawdata.reshape((size,1))
                data = np.unpackbits(rawdata, axis=1)
                newdata = data[...,::-1]
                return newdata.flatten()[:n]
            elif self.header['nbits'] == 2:
                nbyte = int(np.ceil(n/4))
                rawdata = np.fromfile(self._file, dtype='u1', count=nbyte)
                size = rawdata.size
                rawdata = rawdata.reshape((size,1))
                onebit = np.unpackbits(rawdata, axis=1)[...,::-1]
                newdata = np.packbits(onebit.flatten().reshape((-1,2)), axis=1)
                return newdata.flatten()[:n]

            elif self.header['nbits'] == 4:
                rawdata = np.fromfile(self._file, dtype='u1', count=nbyte)
		piece0 = np.bitwise_and(rawdata >> 0x04, 0x0F)
		piece1 = np.bitwise_and(rawdata, 0x0F)
		return np.dstack([piece0, piece1]).flatten()

            elif self.header['nbits'] == 8:
                nbyte = int(np.ceil(n))
                rawdata = np.fromfile(self._file, dtype='u1', count=nbyte)
                #print '*** we are here'
                #print self._file, nbyte
                #print len(rawdata), type(rawdata), rawdata.size
                return rawdata[:n]
            elif self.header['nbits'] == 32:
                nbyte = int(np.ceil(n))
                rawdata = np.fromfile(self._file, dtype='<f4', count=nbyte)
                return rawdata[:n]
        elif self.header['dtype'] == '>u1':
            print "can't handle the dtype", self.header['dtype']
        elif self.header['dtype'] == '<f4' or self.header['nbits'] == 32:
            nbyte = int(np.ceil(n))
            rawdata = np.fromfile(self._file, dtype='<f4', count=nbyte)
            return rawdata[:n]
    
    def close(self):
        self._file.close()
