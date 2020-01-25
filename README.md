# PL, for ENG scroll a little
Program mający zasymulować działanie zmodyfikowanej sieci Barabásiego–Albert. 
Potrzebne komendy znajdziesz poniżej.

# ENG
## Short description
Program written in python (3.7) that simulate modified Barabási–Albert Graph.
I introduce parameter alpha (from 0 to 1), that measure "how much" Barabási–Albert Graph will be mixed with random 
graphs. 
E.g. if alpha=1, then it will be pure BA graph, for alpha=0 it will be pure random graph. And for alpha=0.5 in every 
time step there will be 50% chance that vertex will connect using Preferential Attachment Rule (PAR) - essential part
of BA graph, and 50% chance that will connect to random vertex.


## Commands
`git clone https://github.com/przempol/CustomBarabasiAlbert`

`cd CustomBarabasiAlbert`

`virtualenv venv`

`venv\Scripts\activate`

`pip install -r requirements.txt`

`python main.py [-h] [-c] [-ll] [-ln] m0 alpha desired_size`

|positional arguments | description|
| ------ | ----------- |
| m0              |size of the initial graph (at least 2)
| alpha           |alpha value for modified BA model (from 0 to 1), alpha = 1 mean pure BA, and alpha = 0 mean pure random             
| desired_size    |desired size of the graph at the end


|optional arguments | description|
| ------ | ----------- |
| -h, --help      |show this help message and exit 
| -c, --compare   |compare graph on two chart - log-log scale and log-none scale              
| -ll, --loglog   |plot on log-log scale and perform fit if alpha = 1
| -ln, --lognone  |plot on log-none scale and perform fit if alpha = 0
| |if no optional argument is chosen, then program will calculate time needed to simulate graph

E.g 
* `python main.py 2 1 1000000 -ll`
* `python main.py 3 0 1000000 -ln`
* `python main.py 2 0.5 100000000 -c`

`deactivate`


# # Long description
The Barabási–Albert (BA) model is an algorithm for generating random scale-free networks using a preferential 
attachment mechanism, which means that the more connected a node is, the more likely it is to receive new links.
Nodes with a higher degree have a stronger ability to grab links added to the network.

Program was optimized, by using adjacency list over adjacency matrix. It basically store information about endings of 
every link. It is way better to use than adjacency matrix, because of memory resources. Also one can find easy way to 
transform adjacency list to list of vertex degree - just sum up every appearance in adjacency list for given vertex.

Since Barabási–Albert is very common model, we wanted to generalize it in order to better fit empirical data. Idea was
simple - let's assume that sometimes nodes will connect using PAR, and sometimes just connect random node. So it was the
parameter alpha - which is how big probability is to connect using PAR (in case of alpha=1 it will always connect using
PAR, and vice-versa for alpha=0). 

In the future, there will be more options - since we want to better fit empirical data, we have chosen citation network
as our source of data - ELSEVIER is company specializing in this type of datasets. So at some point there will be 
modules connecting to ELSEVIER using their API etc.