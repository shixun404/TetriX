import redis
import pickle
# r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
# r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
# r = redis.Redis(host='128.55.84.88', port=6379, decode_responses=True)
r = redis.Redis(host='128.55.64.18', port=6379, decode_responses=True)

mydict = {"a":"b", "c":"d"}
p_mydict = pickle.dumps(mydict, protocol=0)
r.set('foo', p_mydict)
# True
p_val = r.get('foo')
val = pickle.loads(str.encode(p_val))
# bar
print(val)

# Import the library
# import redis


# queried_value = None
# try:
#     # Generate the connection
#     r = redis.Redis(host='support-redis.dev.anaconda.com', port=6379)

#     # Set and retrieve the same key
#     r.set('test_key', 'This is a test value for showing redis connectivity')
#     queried_value = r.get('test_key')
# except Exception as e:
#     print(f'Unable to connect or execute commands on Redis server: {e}')

# # Print out queried value
# print(queried_value.decode('utf-8'))