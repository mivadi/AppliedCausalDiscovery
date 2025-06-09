library(devtools)
library(RUcausal)
library(gcimputeR)
library(pcalg)
library(reshape2)
library(ggplot2)
library(mice)
library(micd)
require(sbgcop)
library(reshape2)
library(ggplot2)

print("Download files from https://github.com/cuiruifei/CausalMissingValues")
source('inferCopulaModel.R')
source('gaussCItestLocal.R')

plot_PAG2 <- function (amat, av_skel=1*(amat>0), V = colnames(amat), node_render_info = NULL, 
                       node_size = NULL, R=NULL, track_locations=FALSE, splines=NULL, bbox_new=NULL) 
{
  av_skel <- av_skel * (amat>0)
  if (!all.equal(amat != 0, t(amat) != 0, check.attributes = FALSE)) {
    stop("Input adjacency matrix is not symmetric. Both edge marks must be specified.")
  }
  if (!identical(rownames(amat), colnames(amat))) {
    stop("Input matrix row names must be identical to column names.")
  }
  D <- nrow(amat)
  if (is.null(V)) {
    warning("Using default variable names.")
    V <- as.character(1:D)
  }
  E <- vector("list", length = D)
  names(E) <- V
  for (i in 1:D) E[[V[i]]] <- list(edges = integer(0))
  amat[amat == 1] <- "none"
  amat[amat == 2] <- "open"
  amat[amat == 3] <- "odot"
  ah.list <- list()
  at.list <- list()
  l.names <- list()
  edge.width <- list()
  edge.color <- list()
  edge.lty <- list()
  iE <- 0
  for (i in 1:(D - 1)) {
    for (j in (i + 1):D) {
      if (amat[i, j] != 0) {
        iE <- iE + 1
        E[[V[i]]]$edges <- append(E[[V[i]]]$edges, j)
        E[[V[j]]]$edges <- append(E[[V[j]]]$edges, i)
        ah.list[[iE]] <- amat[j, i]
        at.list[[iE]] <- amat[i, j]
        l.names[[iE]] <- paste0(V[i], "~", V[j])
        edge.width[[l.names[[iE]]]] <- -2/log(.5*av_skel[i,j])  #3*(av_skel[i,j])#-.4)
        # print(edge.width[[l.names[[iE]]]])
        if (!is.null(R)){
          if (R[V[i],V[j]] < 0) {
            if(info$fill[[V[i]]]!=info$fill[[V[j]]]){
              edge.color[[l.names[[iE]]]] <- 'black' #'blue'
            }else{
              edge.color[[l.names[[iE]]]] <- 'grey'
            }
            edge.lty[[l.names[[iE]]]] <- "dashed"
          } else {
            if(info$fill[[V[i]]]!=info$fill[[V[j]]]){
              edge.color[[l.names[[iE]]]] <- 'black' #'blue' 08519C
            }else{
              edge.color[[l.names[[iE]]]] <- 'grey'
            }
            edge.lty[[l.names[[iE]]]] <- "solid"
          }
        }
      }
    }
  }
  # print(edge.width)
  dir.list <- rep(list("both"), length(l.names))
  names(ah.list) <- names(at.list) <- names(dir.list) <- l.names
  if (iE == 0) {
    g <- graph::graphNEL(nodes = V)
  }
  else {
    g <- graph::graphNEL(nodes = V, edgeL = E)
  }
  
  if (!is.null(node_size)) {
    width_vec <- rep(node_size, length(V))
    names(width_vec) <- V
    nAtt <- list(width = width_vec)
    g <- Rgraphviz::layoutGraph(g, nodeAttrs = nAtt)
  }
  else if (iE != 0) {
    g <- Rgraphviz::layoutGraph(g)#, layoutType="neato")
  }
  
  if (!is.null(bbox_new)){
    g@renderInfo@graph$bbox <- bbox_new
  }
  
  if (!is.null(node_render_info)) 
    graph::nodeRenderInfo(g) <- node_render_info
  
  if (!is.null(node_size)) {
    width_vec <- rep(node_size, length(V))
    names(width_vec) <- V
    nAtt <- list(width = width_vec)
  }
  
  if (!is.null(R)){
    graph::edgeRenderInfo(g) <- list(arrowhead = ah.list, arrowtail = at.list, 
                                     dir = dir.list, lwd=unlist(edge.width), col=unlist(edge.color), lty=unlist(edge.lty))
  } else {
    graph::edgeRenderInfo(g) <- list(arrowhead = ah.list, arrowtail = at.list, 
                                     dir = dir.list, lwd=unlist(edge.width))
  }
  
  if (!is.null(splines)){
    for (edge in names(graph::edgeRenderInfo(g)$splines)){
      if (edge %in% names(splines)){
        
        graph::edgeRenderInfo(g)$splines[[edge]] <- splines[[edge]]
        
      } else {
        
        edge_rev <- rev(strsplit(edge, "~")[[1]])
        edge_rev <- paste(edge_rev[[1]], edge_rev[[2]], sep="~")
        
        if (edge_rev %in% names(splines)){
          
          graph::edgeRenderInfo(g)$splines[[edge]] <- splines[[edge_rev]]
          
          arrowhead <- graph::edgeRenderInfo(g)$arrowhead[[edge]]
          arrowtail <- graph::edgeRenderInfo(g)$arrowtail[[edge]]
          
          graph::edgeRenderInfo(g)$arrowhead[[edge]] <- arrowtail
          graph::edgeRenderInfo(g)$arrowtail[[edge]] <- arrowhead
          
        } else {
          print(paste("BezierCurve not yet given of edge", edge))
        }
        
      }
    }
  }
  
  if (!is.null(node_size) | iE != 0) {
    g <- Rgraphviz::renderGraph(g)
  }
  else {
    Rgraphviz::plot(g)
  }
  
  if (track_locations){
    return(list(graph::nodeRenderInfo(g), graph::edgeRenderInfo(g), g@renderInfo@graph$bbox))
  }
}

plot_two_graphs <- function(bccd.av.skel, fci.av.skel, R, bccd.anc, fci.anc, info) {
  # compute the graph that contains all edges
  rownames(bccd.av.skel) <- rownames(R)
  colnames(bccd.av.skel) <- rownames(R)
  av.total <- bccd.av.skel + fci.av.skel[rownames(R), rownames(R)]
  av.total <- av.total - (1*((av.total-1)>0)*(av.total-1))
  anc.total <- 3*((bccd.anc+fci.anc[rownames(R), rownames(R)])!=0)
  
  # plot graph with all edges two times so that box is set well
  out_locations_info <- plot_PAG2(anc.total, av.total, node_render_info = info, R=R, track_locations=TRUE)
  out_locations_info <- plot_PAG2(anc.total, av.total, node_render_info = info, R=R, track_locations=TRUE)
  
  # information about graph that is tracked
  info_new <- out_locations_info[[1]]
  edge_info <- out_locations_info[[2]]
  bbox_new <- out_locations_info[[3]]
  
  rownames(bccd.anc) <- rownames(R)
  colnames(bccd.anc) <- rownames(R)
  
  plot_PAG2(bccd.anc, bccd.av.skel, node_render_info = info_new, R=R, splines=edge_info$splines, bbox_new=bbox_new)
  plot_PAG2(fci.anc, fci.av.skel, node_render_info = info_new, R=R, splines=edge_info$splines, bbox_new=bbox_new)
  
}


getHeatMap <- function(mat, adjmat=F){
  if (adjmat){
    xlabel <- ""
    ylabel <- ""
    settings_side <- theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      panel.grid.major = element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      axis.ticks = element_blank(),
      legend.justification = c(1, 0),
    )
  } else {
    xlabel <- "cause"
    ylabel <- "effect"
    settings_side <- theme(
      # axis.title.x = element_blank(),
      # axis.title.y = element_blank(),
      panel.grid.major = element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      axis.ticks = element_blank(),
      legend.justification = c(1, 0),
    )
  }
  
  melted_cormat <- melt(as.matrix(mat), na.rm = TRUE)
  ggheatmap <- ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) +
    xlab(xlabel) +
    ylab(ylabel) +
    coord_fixed() +
    geom_tile() +
    scale_fill_gradient2(name="ratio", limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 90, vjust = .5, size = 10, hjust = 1)) +
    theme(axis.text.y = element_text(vjust = 0.5, size = 10, hjust = 1)) +
    # geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
    settings_side
  print(ggheatmap)
}



create_boot_datasets_MAR <- function(data, nboot, path_, complete_value_indicator=F){
  N <- nrow(data)
  for(i in 1:nboot){
    # get bootstrapped data set
    bootstrap <- data[sample(N, N, replace=TRUE),]
    if (complete_value_indicator){
      path = paste(path_, "/complete_value_indicator", i,".csv", sep="")
      write.table(1*!is.na(bootstrap), path, sep=",")
    }
    
    # make sure that all columns have distinct values
    while(0%in%(colMeans(bootstrap, na.rm=TRUE) - na.omit(bootstrap)[1,])){
      bootstrap <- data[sample(N, N, replace=TRUE),]
    }
    
    path = paste(path_, "/bootstrap", i,".csv", sep="")
    write.table(bootstrap, path, sep=",")
    
    
    # copula object
    cop.obj <- inferCopulaModel(bootstrap, nsamp = 1000, verb = F)
    # correlation matrix samples
    C_samples <- cop.obj$C.psamp[,, 501:1000]
    # average correlation matrix
    R <- apply(C_samples, c(1,2), mean)
    
    rownames(R) <- names(bootstrap)
    colnames(R) <- names(bootstrap)
    
    path = paste(path_, "/cov_matrix_boot", i,".csv", sep="")
    write.table(R, path, sep=",")
    
  }
} 



averageSkel <- function(outputs, M){
  # outputs must be list
  if (!is.list(outputs)){
    stop("input must be list")
  }
  
  # compute avarage skeleton
  average.skel <- matrix(0,M,M)
  for (alg in outputs){
    if(is.list(alg)){
      average.skel.alg <- matrix(0,M,M)
      for (mat in alg){
        # print(c(ncol(mat),nrow(mat), M))
        average.skel.alg = average.skel.alg + 1*(mat != 0)
      }
      average.skel.alg = average.skel.alg / length(alg)
      # print(length(alg))
    } else {
      average.skel.alg <- 1*(alg != 0)
    }
    average.skel <- average.skel + average.skel.alg
  }
  average.skel = average.skel / length(outputs)
  return(average.skel) 
}

averageArrowType <- function(outputs, M, arrow_type){
  # outputs must be list
  if (!is.list(outputs)){
    stop("input must be list")
  }
  
  # compute avarage skeleton
  average.type <- matrix(0,M,M)
  for (alg in outputs){
    
    if(is.list(alg)){
      average.type.alg <- matrix(0,M,M)
      average.skel.alg <- matrix(0,M,M)
      for (mat in alg){
        average.skel.alg = average.skel.alg + 1*(mat != 0)
        average.type.alg = average.type.alg + 1*(mat == arrow_type)
      }
      average.type.alg = average.type.alg / average.skel.alg
      average.type.alg[which(!is.finite(average.type.alg))] <- 0
    } else {
      average.type.alg <- alg
    }
    average.type <- average.type + average.type.alg
  }
  
  average.type = average.type / length(outputs)
  return(average.type)
}

