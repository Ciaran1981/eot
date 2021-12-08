# encoding: utf-8
"""

DAP dwnloads adapted from CEDA templates with additions and changes by Ciaran Robb

The remainder functions for files ftp by Ciaran Robb

===================


"""

# Import standard libraries
import os
import datetime
import requests
import warnings
# Import third-party libraries
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from contrail.security.onlineca.client import OnlineCaClient
from joblib import Parallel, delayed
from tqdm import tqdm
from urllib.request import urlopen 
from ftplib import FTP


CERTS_DIR = os.path.expanduser('~/.certs')
if not os.path.isdir(CERTS_DIR):
    os.makedirs(CERTS_DIR)

TRUSTROOTS_DIR = os.path.join(CERTS_DIR, 'ca-trustroots')
CREDENTIALS_FILE_PATH = os.path.join(CERTS_DIR, 'credentials.pem')

TRUSTROOTS_SERVICE = 'https://slcs.ceda.ac.uk/onlineca/trustroots/'
CERT_SERVICE = 'https://slcs.ceda.ac.uk/onlineca/certificate/'


def cert_is_valid(cert_file, min_lifetime=0):
    """
    Returns boolean - True if the certificate is in date.
    Optional argument min_lifetime is the number of seconds
    which must remain.
    :param cert_file: certificate file path.
    :param min_lifetime: minimum lifetime (seconds)
    :return: boolean
    """
    try:
        with open(cert_file, 'rb') as f:
            crt_data = f.read()
    except IOError:
        return False

    try:
        cert = x509.load_pem_x509_certificate(crt_data, default_backend())
    except ValueError:
        return False

    now = datetime.datetime.now()

    return (cert.not_valid_before <= now
            and cert.not_valid_after > now + datetime.timedelta(0, min_lifetime))


def setup_credentials():
    """
    Download and create required credentials files.
    Return True if credentials were set up.
    Return False is credentials were already set up.
    :param force: boolean
    :return: boolean
    """

    # Test for DODS_FILE and only re-get credentials if it doesn't
    # exist AND `force` is True AND certificate is in-date.
    if cert_is_valid(CREDENTIALS_FILE_PATH):
        print('[INFO] Security credentials already set up.')
        return False

    
    username = os.environ['CEDA_USERNAME']
    password = os.environ['CEDA_PASSWORD']

    onlineca_client = OnlineCaClient()
    onlineca_client.ca_cert_dir = TRUSTROOTS_DIR

    # Set up trust roots
    trustroots = onlineca_client.get_trustroots(
        TRUSTROOTS_SERVICE,
        bootstrap=True,
        write_to_ca_cert_dir=True)

    # Write certificate credentials file
    key_pair, certs = onlineca_client.get_certificate(
        username,
        password,
        CERT_SERVICE,
        pem_out_filepath=CREDENTIALS_FILE_PATH)

    print('[INFO] Security credentials set up.')
    return True

def setup_sesh(user, password):
    
    """
    setup user/pass
    
    Parameters
    ----------
    
    user: string
        CEDA username
    
    password: string
        CEAD password

    """
    os.environ['CEDA_USERNAME'] = user
    os.environ['CEDA_PASSWORD'] = password


def dload(file_url, folder, method='requests'):
    """
    Download a file from the CEDA archive
    
    Parameters
    ----------
    
    file_url: string
        URL to a NetCDF4 opendap end-point.
    
    folder: string
        the dir in which to download the file
        
    Returns
    -------
    path of file

    """

    try:
        setup_credentials()
    except KeyError:
        print("CEDA_USERNAME and CEDA_PASSWORD environment variables required")
        return

    # Download file to current working directory
    # requests is a bit unreliable with the nextmap data
    if method != 'requests':
        response = urlopen(file_url)  
        finalrep = response.read()
    else:
        response = requests.get(file_url, cert=(CREDENTIALS_FILE_PATH), verify=False)
        finalrep = response.content
        
    filename = file_url.rsplit('/', 1)[-1]
    final = os.path.join(folder, filename)
    
    # dwnld
    with open(final, 'wb') as file_object:
        file_object.write(finalrep)
    # some problem is occurring with a repeat download of the first file
    del response, finalrep, filename, file_object
    
    return final



def dloadbatch(urls, folder, para=False, nt=-1, method='requests'):
    
    """
    Download a batch of files from the CEDA archive
    
    Parameters
    ----------
    
    urls: string
        URLs to a NetCDF4s opendap end-point.
    
    folder: string
        the dir in which to download the file
        
    para: bool
        whether to process in parallel def. False
    
    method: string
        requests or urllib
    
    Returns
    -------
    list of file paths
    """

    with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if para == True:
                paths = Parallel(n_jobs=nt, verbose=2)(delayed(dload)(
                        u, folder, method=method) for u in urls)
            else:
                paths = [dload(u, folder) for u in tqdm(urls)]
    return paths
    

def dtmftp(user, passwd, ftp_path, main_dir):
    
    """
    Download the files for a nextmap dtm folder
    
    Parameters
    ----------
    
    user: string
            CEDA usernm
            
    passwd: string
            CEDA passwd
    
    ftp_path:string
            the ftp path to dir containing the DTM eg
            'neodc/nextmap/by_tile/sh/sh60/dtm/sh60dtm/'
    
    main_dir: string
                the local dir in which all subdirs and files are dwnlded
            
    
    """
    # as it is parallel required ....
    ftp = FTP("ftp.ceda.ac.uk", "", "")
    ftp.login(user=user, passwd=passwd)
    # navigate to the dir
    ftp.cwd(ftp_path)
    # I hate ESRI
    esri_types = ['dblbnd.adf', 'hdr.adf', 'prj.adf',
              'sta.adf', 'w001001.adf', 'w001001x.adf']
    # outdir in which the dinosaur format goes 
    dirname = os.path.join(main_dir, ftp_path.split(sep="/")[6])
    if not os.path.isdir(dirname):
        os.mkdir(dirname) 
    # loop through the files and write to disk
    for e in esri_types:
        localfile = os.path.join(dirname, e)
        with open(localfile, "wb") as lf:
            ftp.retrbinary('RETR ' + e, lf.write, 1024)
    # why is this not quitting.....
    ftp.quit()
    return dirname

def dtmftp_mt(user, passwd, ftplist, main_dir):
    
    """
    Download the files for a nextmap dtm folder
    
    Parameters
    ----------
    
    user: string
            CEDA usernm
            
    passwd: string
            CEDA passwd
    
    ftplist: list of strings
            a list containing ftp paths like below
            ['neodc/nextmap/by_tile/sh/sh60/dtm/sh60dtm/']
    
    main_dir: string
                the local dir in which all subdirs and files are dwnlded
            
    
    """
    
    out = Parallel(n_jobs=8, verbose=2)(delayed(dtmftp)(
            user, passwd, f, main_dir) for f in ftplist)
    
    return out


