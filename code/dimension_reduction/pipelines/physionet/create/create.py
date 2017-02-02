
def create_hrd_uns(P):    
    
    ncomponents = "        n_components: " + str(P) + "\n"      
    label       = "label: 'hrd-uns (" + str(P) +  ") + mdm'"
    
    lines = ["imports:\n",
             "    pyriemann.estimation:\n",
             "        - Covariances\n",
             "    pyriemann.classification:\n",
             "        - MDM\n",
             "    utilities.dim_reduction:\n",
             "        - RDR\n\n",
             "pipeline:\n",
             "    - Covariances:\n"
             "        estimator: 'oas'\n",        
             "    - RDR:\n",
             ncomponents,
             "        method: 'harandi-uns'\n",
             "    - MDM:\n",
             "        metric: 'riemann'\n\n",         
             label
             ]     
             
    return lines                 
    
def create_mmx(P):
    
    ncomponents = "        n_components: " + str(P) + "\n"  
    label       = "label: 'minmax (" + str(P) + ") + mdm'"               
    
    lines = ["imports:\n",
             "    pyriemann.estimation:\n",
             "        - Covariances\n",
             "    pyriemann.classification:\n",
             "        - MDM\n",
             "    utilities.dim_reduction:\n",
             "        - RDR\n\n",
             "pipeline:\n",
             "    - Covariances:\n"
             "        estimator: 'oas'\n",        
             "    - RDR:\n",
             ncomponents,
             "        method: 'minmax'\n",             
             "        alpha: 1.0\n" ,             
             "    - MDM:\n",
             "        metric: 'riemann'\n\n",         
             label
             ]     
             
    return lines 

def create_nrme(P):
    
    ncomponents = "        n_components: " + str(P) + "\n"  
    label       = "label: 'nrme (" + str(P) + ") + mdm'"               
    
    lines = ["imports:\n",
             "    pyriemann.estimation:\n",
             "        - Covariances\n",
             "    pyriemann.classification:\n",
             "        - MDM\n",
             "    utilities.dim_reduction:\n",
             "        - RDR\n\n",
             "pipeline:\n",
             "    - Covariances:\n"
             "        estimator: 'oas'\n",        
             "    - RDR:\n",
             ncomponents,
             "        method: 'nrme'\n",                        
             "    - MDM:\n",
             "        metric: 'riemann'\n\n",         
             label
             ]     
             
    return lines    

def create_covpca(P):
    
    ncomponents = "        n_components: " + str(P) + "\n"  
    label       = "label: 'covpca (" + str(P) + ") + mdm'"               
    
    lines = ["imports:\n",
             "    pyriemann.estimation:\n",
             "        - Covariances\n",
             "    pyriemann.classification:\n",
             "        - MDM\n",
             "    utilities.dim_reduction:\n",
             "        - RDR\n\n",
             "pipeline:\n",
             "    - Covariances:\n"
             "        estimator: 'oas'\n",        
             "    - RDR:\n",
             ncomponents,
             "        method: 'covpca'\n",                        
             "    - MDM:\n",
             "        metric: 'riemann'\n\n",         
             label
             ]     
             
    return lines  

def create_csp(P):
    
    ncomponents = "        nfilter: " + str(P) + "\n"  
    label       = "label: 'csp (" + str(P) + ") + mdm'"               
    
    lines = ["imports:\n",
             "    pyriemann.estimation:\n",
             "        - Covariances\n",
             "    pyriemann.classification:\n",
             "        - MDM\n",
             "    pyriemann.spatialfilters:\n",
             "        - CSP\n\n",
             "pipeline:\n",
             "    - Covariances:\n"
             "        estimator: 'oas'\n",        
             "    - CSP:\n",
             ncomponents,
             "        log: False\n",                        
             "    - MDM:\n",
             "        metric: 'riemann'\n\n",         
             label
             ]     
             
    return lines    
         
P  = [4,8,12,16,20,24,28,32,36]
for Pi in P:
     
    filename = 'pipeline_2_hrd-uns_p' + "{0:02d}".format(Pi) + '_mdm.yaml'
    f = open(filename, 'w') 
    for line in create_hrd_uns(Pi):
        f.write(line)   
    f.close()       

    filename = 'pipeline_3_mmx_p' + "{0:02d}".format(Pi) + '_mdm.yaml'
    f = open(filename, 'w') 
    for line in create_mmx(Pi):
        f.write(line)   
    f.close() 
    
    filename = 'pipeline_4_nrme_p' + "{0:02d}".format(Pi) + '_mdm.yaml'
    f = open(filename, 'w') 
    for line in create_nrme(Pi):
        f.write(line)   
    f.close()     
               
    filename = 'pipeline_5_covpca_p' + "{0:02d}".format(Pi) + '_mdm.yaml'
    f = open(filename, 'w') 
    for line in create_covpca(Pi):
        f.write(line)   
    f.close() 

    filename = 'pipeline_6_csp_p' + "{0:02d}".format(Pi) + '_mdm.yaml'
    f = open(filename, 'w') 
    for line in create_csp(Pi):
        f.write(line)   
    f.close()       
        








