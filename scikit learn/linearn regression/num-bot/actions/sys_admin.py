import paramiko

hostname = 'molynew.com'
port = 22
user= 'ubuntu'
key_file= '/home/onesmus/keys/hosts.pem'

def connect_to_host(hostname,port,user,key_file):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=hostname,port=port,username=user,key_filename=key_file)
    except Exception as e:
        raise e
    return client

def task_list_files(cmd):
    try:
        client = connect_to_host(hostname,port,user,key_file)
        stdin, stdout, stderr = client.exec_command(cmd)
        data = stdout.readlines()
        client.close()
    except Exception as e:
        raise e
    return data