## The data generation process

```mermaid
flowchart TB   
	subgraph culture[Zebrafish culture]
        N_pop((n)) --- batch
    	population(Zebrafish population) --- batch(Batch)
    end
        
    subgraph experiment
    	direction TB
    	batch --> organisms[Test organisms]
        organisms ---> |select & incubate| transfer{Transfer}
        
        stock((Stock solution)) ---- |preparation error| dilution{Serial dilution} & dilution_2{Serial dilution}

        
        vol((Vol. transfer)) --> |50 µl| transfer
        
        N_per_vial((n ZFE per vial)) --- |n=3| transfer
        volume((Volume)) --- |6 ml| transfer_exposure & transfer_exposure_2
        
        transfer ----> |pipetting error| R1[Vial 1] & R2[Vial 2] & R3[Vial 3] & R4[...] & RN[Vial N]
        transfer ----> |pipetting error| R1_2[Vial 1] & R2_2[...] & RN_2[Vial N]
        
        subgraph treatment_1[Treatment 1]
            subgraph solution_prep
                n_steps_t1((steps)) --- |pipetting error| dilution
                dilution --- exposure((Exposure solution))
            end
            

            transfer_exposure --- R1 & R2 & R3 & R4 & RN
            subgraph replicates
                exposure --> transfer_exposure{Transfer}
        	    R1 & R2 & R3 & R4 & RN --> pool_1[Pooled samples]	
            end
        end
        
        subgraph treatment_2[Treatment 2]
            subgraph solution_prep_2
                n_steps_t2((steps)) --- |pipetting error| dilution_2
                dilution_2 --- exposure_2((Exposure solution))
            end
            transfer_exposure_2 --- R1_2 & R2_2 & RN_2
            subgraph replicates_2
                exposure_2 --> transfer_exposure_2{Transfer}
        	    R1_2 & R2_2 & RN_2 --> pool_2[Pooled samples]	
            end
        end
                
        N_sample((n sample)) --- sample_1 & sample_2
        pool_1 --> sample_1[Sample T1]
        pool_2 --> sample_2[Sample T2]

    end
    
    sample_1 --> OR{{OR}}
    
        OR --> wash_rinse{wash and rinse}
    subgraph internal_concentration[Internal concentration]
        wash_rinse --> freeze{freeze}
        freeze --> homogenized[Homogenized sample]
        dilution_meas_vol((Vol.)) --> dilution_meas
        homogenized --> dilution_meas{Dilution ?}
        error((Error)) --> measure
        dilution_meas --> measure{LCMS measurement}
        measure --> value
    end
    
    
        OR --> process{wash and rinse}
    subgraph gene_expression[Gene Expression]
        process --> freeze_rna{freeze}
        freeze_rna --> homogenized_rna[Homogenized sample]
        homogenized_rna --> extract{extract}
        extract --> OR_RNA_METHOD{{Method}}
        OR_RNA_METHOD --> microarray
        OR_RNA_METHOD --> RNA-seq
        OR_RNA_METHOD --> qpcr
        subgraph RNA_method_microarray[Microarray]
            microarray
        end
        subgraph RNA_method_rnaseq[RNA sequencing]
            RNA-seq
        end
        subgraph RNA_method_qpcr[qPCR]
            qpcr
        end
    end


```

### Schüttler data

gene expression
The LC25, modeled from experimental observations served as highest and the LC0.5 as lowest exposure concentration with 6 equal dilution steps in between, with dilution steps 1,2,4 and 6 chosen for exposure.

Parameters

+ n_steps_t1 = 1
+ n_steps_t2 = 2
+ n_steps_t3 = 4
+ n_steps_t4 = 6