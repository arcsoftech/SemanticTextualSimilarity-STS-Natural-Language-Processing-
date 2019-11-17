def cosine_simlarity(v1,v2):
        # cosine formula 
        c=0 
        for i in range(len(v1)): 
                c+= v1[i]*v2[i] 
        cosine = c / float((sum(v1)*sum(v2))**0.5) 
        return cosine
        