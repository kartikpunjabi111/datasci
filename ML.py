import mysql.connector

class ML_DB():
    
    def __init__(self):
        self.cursor=None
        self.conn=None
        
    def connect_to_db(self,host=None,user=None,password=None,database=None):
        '''
        To connect to a Database on MySQL server
        :params -host=host address or IP address machine hosting MySQL engine, STRING
                -user=user name, STRING
                -password=password, STRING
                -database=database's name, STRING
                
        :return -connection object of MySQL connector for python
                -cursor
        '''
        self.conn=mysql.connector.connect(host=host, user=user, password=password, database=database)
        self.cursor=self.conn.cursor()
        print('connected ,ID', self.conn.connection_id)
        
        return self.conn, self.cursor
    
    def close_connection(self):
        '''
        To close the server connection connected to this class attributed cursor and database
        '''
        self.cursor.close()
        self.conn.close()
        print('connection to ' + str(self.conn.connection_id) + ' is closed')
        
    def adapt_array(self,array):
        """
        Using the numpy.save function to save a binary version of the array,
        and BytesIO to catch the stream of data and convert it into a BLOB.
        
        :params -array=type array
        
        :return -converted arrray in BLOB datatype
                
        """
        import io
        import array,numpy
        out = io.BytesIO()
        numpy.save(out, array)
        out.seek(0)
        
        return out.read()

    def convert_array(self,blob):
        """
        Using BytesIO to convert the binary version of the array back into a numpy array.
        :params -array=type bytes encoded STRING
        
        :return -converted bytes encoded STRING to NUMERICAL NUMPY array
        """
        import io
        import array,numpy
        out = io.BytesIO(blob)
        out.seek(0)

        return numpy.load(out)
    
