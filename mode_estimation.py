# mode_estimation.py
# Traffic mode estimation and activity area recognition
# Sirapat Na Ranong , July 2017

# Steps:
# 1 Read files from hdfs
# 2 Calculate time step
# 3 Calculate distance and speed
# 4 Remove noises
# 5 Estimate the mode
# 6 Improve the estimated mode
# 7 Find activity area and group the nearby points
# 8 Write to hdfs

from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.functions import lag, when, col
from pyspark.sql.functions import cos, sin, asin, sqrt, toRadians

spark = SparkSession.builder.appName("TrafficModeEstimation").getOrCreate()

df = spark.read.csv("hdfs:///user/root/cmp_traj/*", header= True) # read input files

df = df.drop("ts_week","ts_weekday","hum","irtemp","light","noise","press","temp","period","week_day","id_week_day","mode","cmode")
#from pyspark.sql.functions import input_file_name, regexp_replace
#df = df.withColumn("filename", regexp_replace(input_file_name(), ".*/|.csv", "") ) # get the input file name
df = df.withColumn("id", df.nid)

w = Window.partitionBy("nid").orderBy("timestamp")

# time_step = (timestamp of current row - previous row)
df = df.withColumn("time_step", (df.timestamp - lag(df.timestamp, 1).over(w)).cast("int") )

# if 20 <= time_step <= 60 seconds then speed_step = (steps of current row - previous row) / time_step
df = df.withColumn("speed_step", when((df.time_step >= 20) & (df.time_step <= 60) ,
									((df.steps - lag(df.steps,1).over(w)) / df.time_step).cast("decimal(10,2)") ) )

# Haversine algorithm  https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
def haversine(lon1, lat1, lon2, lat2):
	lon1, lat1, lon2, lat2 = map(toRadians, [lon1.cast("float"), lat1.cast("float"), lon2.cast("float"), lat2.cast("float")])
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
	c = 2 * asin(sqrt(a))
	m = 6367000 * c
	return m.cast("decimal(10,2)") # meters

# if 20 <= time_step <= 60 seconds then speed_distance = distance calculated by Haversine algorithm / time_step
df = df.withColumn("speed_distance", when((df.time_step >= 20) & (df.time_step <= 60) ,
										((haversine(df.lon, df.lat, lag(df.lon,1).over(w), lag(df.lat,1).over(w))) / df.time_step).cast("decimal(10,2)") ) )

df = df.filter((df.time_step >= 20) & (df.time_step <= 60))
df = df.withColumn("time_step", (df.timestamp - lag(df.timestamp, 1).over(w)).cast("int") ) # new time_step
df = df.filter((df.speed_step >= 0) & (df.speed_step <= 5)).filter(df.speed_distance <= 55) # Remove noises

df = df.withColumn("mode", when(df.speed_distance == 0, 1) 											 # WALKING (almost static posiibly around activity point)
							.when(df.speed_distance >= 4, 0)										 # TRANSIT
							.when((df.speed_step <= 0.3) & (df.speed_distance > 1), 0) 				 # TRANSIT (includes traffic jam / red light)
							.when((df.speed_step <= 0.3) & (df.speed_distance <= 1), 1) 			 # WALKING (slow)
							.when((df.speed_step > 0.3) & (df.speed_distance/df.speed_step >= 4), 0) # TRANSIT (includes some steps)
							.when((df.speed_step > 0.3) & (df.speed_distance/df.speed_step < 4), 1)  # WALKING (quicker)
							.otherwise(1) )

df = df.withColumn("mode", when((df.time_step + lag(df.time_step, -1).over(w) <= 300) &  # In time window = 5 minutes of current and next row
								(df.mode != lag(df.mode,-1).over(w)) &				 	 # If current mode != next mode
								(df.mode != lag(df.mode,1).over(w)),				 	 # and current mode != previous mode
								lag(df.mode,1).over(w) ) 							 	 # then change the mode
								.otherwise(df.mode) ) 								 	 # else retain the same mode

df = df.withColumn("mode", when((df.time_step + lag(df.time_step, -1).over(w) <= 300) &  ####################################
								(df.mode != lag(df.mode,-1).over(w)) &			 	 	 # 		Repeat above function		#
								(df.mode != lag(df.mode,1).over(w)) ,				 	 #		to improve mode      		#
								lag(df.mode,1).over(w) ) 							 	 # 		 							#
								.otherwise(df.mode) ) 								 	 ####################################

df = df.withColumn("mode", when((df.time_step+lag(df.time_step,1).over(w) <= 300) & 	 # In time window = 5 minutes of current and previous row
								(df.mode == lag(df.mode,1).over(w)) &					 # If current mode = previous mode
								(df.mode != lag(df.mode,-1).over(w)) &					 # and current mode != next mode
								(lag(df.mode,-1).over(w) == lag(df.mode,2).over(w)),	 # and next mode = previous of previous mode
								lag(df.mode,-1).over(w) )								 # then change mode = next mode
							.when((df.time_step+lag(df.time_step,-1).over(w) <= 300) &	 # In time window = 5 minutes of current and next row
								(df.mode == lag(df.mode,-1).over(w)) &  				 # If cureent mode = next mode
								(df.mode != lag(df.mode,1).over(w)) &					 # and current mode != previous mode
								(lag(df.mode,1).over(w) == lag(df.mode,-2).over(w)) ,	 # and previous mode = next of next mode
								lag(df.mode,1).over(w) )								 # then mode = previous mode
							.otherwise(df.mode) )										 # else retain the same mode

# Activity area recognition
# Mark as tempActivityPoint if the speed is 0
df = df.withColumn("tempActivityPoint", when((df.speed_distance == 0) & (df.speed_step == 0), 1).otherwise(0))

w = Window.partitionBy("nid","tempActivityPoint").orderBy("timestamp")

# Mark as activityPoint if its distance from another tempActivityPoint is more than 300 meters
df = df.withColumn("activityPoint", when(df.tempActivityPoint == 1,
									when((haversine(df.lon, df.lat, lag(df.lon,-1).over(w), lag(df.lat,-1).over(w)) >= 300) , 1 ).otherwise(0) )
									.otherwise(0) )

df = df.filter(~((df.mode == 1) & (df.activityPoint == 0)))	# remove walking point around activity point
df = df.drop("steps","timestamp","acc","tempActivityPoint")
df.write.partitionBy("id").format('csv').save("hdfs:///user/root/mode_cmp_traj",header=True) #write to a folder in hdfs
