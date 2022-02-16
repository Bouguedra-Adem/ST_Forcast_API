import geohash
import numpy as np

# Cette classe définit les fonctions du système géohash
class Hash:
    def __init__(self, precision):
        self.precision = precision
        self._base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
        self._base32_map = {}
        for i in range(len(self._base32)):
            self._base32_map[self._base32[i]] = i
        del i


    def geohashFunction(self,x,y):
       return geohash.encode(x,y,self.precision)


    def decode_c2i(self,hashcode):
        lon = 0
        lat = 0
        bit_length = 0
        lat_length = 0
        lon_length = 0
        for i in hashcode:
            t = self._base32_map[i]
            if bit_length % 2 == 0:
                lon = lon << 3
                lat = lat << 2
                lon += (t >> 2) & 4
                lat += (t >> 2) & 2
                lon += (t >> 1) & 2
                lat += (t >> 1) & 1
                lon += t & 1
                lon_length += 3
                lat_length += 2
            else:
                lon = lon << 2
                lat = lat << 3
                lat += (t >> 2) & 4
                lon += (t >> 2) & 2
                lat += (t >> 1) & 2
                lon += (t >> 1) & 1
                lat += t & 1
                lon_length += 2
                lat_length += 3

            bit_length += 5

        return (lat, lon, lat_length, lon_length)


    def encode_i2c(self,lat, lon, lat_length, lon_length):
        precision = int((lat_length + lon_length) / 5)
        if lat_length < lon_length:
            a = lon
            b = lat
        else:
            a = lat
            b = lon

        boost = (0, 1, 4, 5, 16, 17, 20, 21)
        ret = ''
        for i in range(precision):
            ret += self._base32[(boost[a & 7] + (boost[b & 3] << 1)) & 0x1F]
            t = a >> 3
            a = b >> 2
            b = t

        return ret[::-1]


    def neighbors(self,hashcode, S):
        (lat, lon, lat_length, lon_length) = self.decode_c2i(hashcode)
        ret = []
        rang = S // 2
        tab = []
        tab2 = []
        tab3 = []

        for i in range(1, rang + 1):
            tab.append(lat + rang + 1 - i)
            tab3.append(lat - i)
        for i in range(-rang, rang + 1):
            tab2.append(lon + i)

        for tlat in tab:
            if not tlat >> lat_length:
                for tlon in tab2:
                    ret.append(self.encode_i2c(tlat, tlon, lat_length, lon_length))

        tlat = lat
        for tlon in tab2:
            code = self.encode_i2c(tlat, tlon, lat_length, lon_length)
            if code:
                ret.append(code)

        for tlat in tab3:
            if tlat >= 0:
                for tlon in tab2:
                    ret.append(self.encode_i2c(tlat, tlon, lat_length, lon_length))

        return np.array(ret).reshape(S, S)