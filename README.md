# suspicious
A Python clone of https://github.com/apache/incubator-spot/ spot-ml module

As we known, spot-ml module provided three kind of network anomaly detection, Netflow, DNS, and Proxy.

spot-ml adopted topic module for Netflow anomaly detection.
And I am interested about how to map a problem from anomaly detection to topic module.

Wrote this project just for fun.

## Prepare data for input

**Netflow Data**

Netflow data should be a csv file and include the following schema:

- time: String
- year: Double
- month: Double
- day: Double
- hour: Double
- minute: Double
- second: Double
- time of duration: Double
- source IP: String
- destination IP: String
- source port: Double
- dport: Double
- proto: String
- flag: String
- fwd: Double
- stos: Double
- ipkt: Double.
- ibyt: Double
- opkt: Double
- obyt: Double
- input: Double
- output: Double
- sas: String
- das: Sring
- dtos: String
- dir: String
- rip: String

## Output data

The output data should be a csv file include all input columns and two additional columns, estimated probabilities and v.

- probability: Double
- v: Bool
